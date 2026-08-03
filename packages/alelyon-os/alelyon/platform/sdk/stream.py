"""EngineStream — live telemetry from a running Alelyon engine over ZMQ.

The engine broadcasts single-frame JSON messages (types: health, pnl, position,
signal, order, feature, tick, news) on a PUB socket, by default 127.0.0.1:5555.
This is an application-level read-only SUBSCRIBER: it sends no engine command.
Its TCP/ZMQ connection still consumes publisher and host resources.

    from alelyon.platform.sdk import EngineStream
    with EngineStream() as es:
        for msg in es.messages(types={"health", "pnl"}, timeout_s=30):
            print(msg["type"], msg["data"])

`messages()` returns after `timeout_s` elapses with no message it would YIELD.
That clock runs on messages passing the type filter, not on raw traffic — a busy
engine publishing only types you did not ask for still ends the loop instead of
hanging forever. With no filter the two are the same thing.

The publisher sends no topic frames, so filtering is client-side by msg["type"].
Remote unauthenticated plaintext endpoints are refused by default; prefer a
trusted tunnel rather than opting into disclosure with allow_insecure_remote.

Requires the `stream` extra: pip install "alelyon-os[stream]"
"""
from __future__ import annotations

import ipaddress
import json
import math
import time
from typing import Iterator, Optional, Set

from alelyon.platform.sdk._json import (
    json_nesting_within_limit,
    parse_bounded_int,
    parse_finite_float,
    reject_duplicate_pairs,
    reject_nonfinite_constant,
)


DEFAULT_MAX_FRAME_BYTES = 1024 * 1024
DEFAULT_RECEIVE_HWM = 1000


def _loopback_endpoint(host: str) -> tuple[bool, str]:
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty hostname or IP address")
    normalized = host.strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if any(character in normalized for character in "/?#@"):
        raise ValueError("host must not contain a URL scheme, path, or userinfo")
    lower = normalized.rstrip(".").lower()
    is_loopback = lower == "localhost"
    try:
        address = ipaddress.ip_address(lower)
    except ValueError:
        if ":" in normalized:
            raise ValueError("host is not a valid IP address") from None
    else:
        is_loopback = address.is_loopback
    endpoint_host = f"[{normalized}]" if ":" in normalized else normalized
    return is_loopback, endpoint_host


class EngineStream:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, *,
                 allow_insecure_remote: bool = False,
                 max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
                 receive_hwm: int = DEFAULT_RECEIVE_HWM,
                 enable_curve: bool = True) -> None:
        loopback, endpoint_host = _loopback_endpoint(host)
        if not isinstance(allow_insecure_remote, bool):
            raise TypeError("allow_insecure_remote must be a bool")
        if not loopback and not allow_insecure_remote:
            raise ValueError(
                "refusing unauthenticated plaintext telemetry from a remote "
                "host; use a trusted tunnel or pass allow_insecure_remote=True "
                "after accepting the disclosure risk"
            )
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65535
        ):
            raise ValueError("port must be an integer from 1 through 65535")
        if (
            isinstance(max_frame_bytes, bool)
            or not isinstance(max_frame_bytes, int)
            or max_frame_bytes <= 0
        ):
            raise ValueError("max_frame_bytes must be a positive integer")
        if (
            isinstance(receive_hwm, bool)
            or not isinstance(receive_hwm, int)
            or receive_hwm <= 0
        ):
            raise ValueError("receive_hwm must be a positive integer")
        try:
            import zmq
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise ModuleNotFoundError(
                "EngineStream needs pyzmq — install the extra: "
                'pip install "alelyon-os[stream]"') from exc
        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock = None
        try:
            self._sock = self._ctx.socket(zmq.SUB)
            self._sock.setsockopt(zmq.SUBSCRIBE, b"")
            self._sock.setsockopt(zmq.RCVTIMEO, 250)
            self._sock.setsockopt(zmq.RCVHWM, receive_hwm)
            self._sock.setsockopt(zmq.MAXMSGSIZE, max_frame_bytes)
            # The engine publishes over a CURVE server socket. An unarmed SUB
            # connects successfully and then receives nothing at all, which is
            # indistinguishable from a stopped engine — so arm from the local
            # sealed identity, and if that is impossible say why instead of
            # handing back a stream that will never yield.
            if enable_curve:
                self._arm_curve()
            self._sock.connect(f"tcp://{endpoint_host}:{port}")
            self._max_frame_bytes = max_frame_bytes
        except Exception:  # noqa: BLE001 - never leak the context on a bad ctor
            if self._sock is not None:
                self._sock.close(linger=0)
            self._ctx.term()
            raise

    def _arm_curve(self) -> None:
        """Attach this machine's CURVE client identity, or refuse to construct.

        Imported lazily and from `curve_identity` rather than the Qt IPC module:
        an SDK that dragged PyQt6 in to read a telemetry stream would be a poor
        client. A checkout without the gateway package still works — the import
        failure is reported as the same actionable error rather than a traceback
        from an unrelated layer.
        """
        try:
            from alelyon.platform.gateway.curve_identity import arm_client_socket
        except ImportError as exc:                  # pragma: no cover
            raise PermissionError(
                "EngineStream cannot reach the local engine transport identity "
                "and the telemetry socket is encrypted; pass enable_curve=False "
                "only against a publisher you know is plaintext") from exc
        if not arm_client_socket(self._sock, "EngineStream subscription"):
            raise PermissionError(
                "EngineStream is not enrolled for encrypted engine telemetry. "
                "The engine mints the trust record when it starts, and only the "
                "same OS user can read it. Start the engine, run as that user, "
                "or pass enable_curve=False for a plaintext publisher."
            )

    def messages(self, *, types: Optional[Set[str]] = None,
                 timeout_s: float = 30.0,
                 max_messages: Optional[int] = None) -> Iterator[dict]:
        """Yield engine messages, optionally filtered by type. Returns after
        `timeout_s` with nothing yielded, or after `max_messages` yielded."""
        if isinstance(types, str):
            raise TypeError(
                'types must be a set of type names, not a string — '
                f'pass {{"{types}"}} rather than "{types}"')
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise TypeError("timeout_s must be a finite positive number")
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be a finite positive number")
        if max_messages is not None and (
            isinstance(max_messages, bool)
            or not isinstance(max_messages, int)
            or max_messages <= 0
        ):
            raise ValueError("max_messages must be a positive integer or None")
        wanted = None if types is None else set(types)
        if wanted is not None and any(
            not isinstance(item, str) or not item for item in wanted
        ):
            raise ValueError("types must contain only non-empty strings")
        yielded = 0
        last_yield = time.monotonic()
        while True:
            # Checked every iteration, NOT only when the socket times out: a
            # publisher flooding types we filter out never raises Again, so a
            # timeout tested only on that branch would never fire.
            if time.monotonic() - last_yield >= timeout_s:
                return
            try:
                frame = self._sock.recv()
            except self._zmq.Again:
                continue
            if len(frame) > self._max_frame_bytes:
                continue
            try:
                if not json_nesting_within_limit(frame):
                    continue
                msg = json.loads(
                    frame.decode("utf-8"),
                    parse_constant=reject_nonfinite_constant,
                    parse_float=parse_finite_float,
                    parse_int=parse_bounded_int,
                    object_pairs_hook=reject_duplicate_pairs,
                )
            except (ValueError, UnicodeDecodeError, RecursionError):
                # A malformed frame is the publisher's problem, not a reason to
                # tear down the subscriber mid-session.
                continue
            if not isinstance(msg, dict):
                continue
            message_type = msg.get("type")
            if not isinstance(message_type, str) or not message_type:
                continue
            if wanted is not None and message_type not in wanted:
                continue
            yielded += 1
            # Record the event before transferring control. Consumer work then
            # counts toward the documented "since last yield" idle clock.
            last_yield = time.monotonic()
            yield msg
            if max_messages is not None and yielded >= max_messages:
                return

    def close(self) -> None:
        if self._sock is None:
            return
        sock, self._sock = self._sock, None
        try:
            sock.close(linger=0)
        finally:
            self._ctx.term()

    def __enter__(self) -> "EngineStream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

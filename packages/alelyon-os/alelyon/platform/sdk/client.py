"""AlelyonClient — the Python client for the Alelyon HTTP API.

    from alelyon.platform.sdk import AlelyonClient
    c = AlelyonClient()                              # http://127.0.0.1:8710
    c = AlelyonClient("https://host:8710", api_key="...")

    c.health()
    c.bars("AAPL", lookback_days=90)                 # certified OHLCV + certificate
    c.quote("SPY"); c.quotes(["AAPL", "MSFT"], include_certs=True)
    c.fred("DGS10")
    c.certificate("bars", "AAPL", "1d")
    c.rates(); c.vol("SPY"); c.rotation(); c.crash(); c.breadth()
    c.analyst("MSFT"); c.screener(["AAPL","MSFT"]); c.pulse()
    c.answer("3-month correlation of SPY and TLT")
    c.certified_pubkey(); c.certify(program="...")   # signed CNE
    c.engine_status()

Every method returns the endpoint's decoded JSON. Most return a dict; `screener`
returns a LIST of board tiles. Any non-2xx response raises ApiError carrying
`.status_code` and the server's `detail`; a 2xx whose body is not JSON raises
ApiError too, rather than a bare json decode error escaping the SDK.

Redirects are followed by default, but only on the same origin or for a
same-host HTTP-to-HTTPS upgrade. Cross-origin redirects and TLS downgrades are
refused before a second request is sent. Pass follow_redirects=False to see any
redirect as ApiError instead.

The transport kwarg accepts any SYNC httpx transport (e.g. httpx.MockTransport
for tests, or httpx.WSGITransport). NOTE: httpx.ASGITransport is async-only and
will NOT work with this sync client — for in-process ASGI testing, run the app
under a real loopback server.
"""
from __future__ import annotations

import ipaddress
import math
import time
from typing import Any, Iterable, Optional
from urllib.parse import quote as _urlquote

import httpx

from alelyon.platform.sdk._json import (
    json_nesting_within_limit,
    parse_bounded_int,
    parse_finite_float,
    reject_duplicate_pairs,
    reject_nonfinite_constant,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8710"

#: The server caps a batched quote request at this many symbols (and silently
#: drops the rest), so the client refuses to build a request it knows will be
#: truncated. Chunk larger universes yourself.
MAX_QUOTE_SYMBOLS = 100

# Market-data responses can legitimately contain years of bars, but the SDK
# must not let a broken proxy or hostile endpoint grow memory without bound.
DEFAULT_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_REDIRECTS = 5
MIN_API_KEY_LENGTH = 32
MAX_API_KEY_LENGTH = 512


def _is_loopback_host(host: str) -> bool:
    """Return whether *host* is an explicit loopback name or address.

    Hostnames are deliberately not resolved: DNS-based loopback classification
    would make the plaintext-key guard vulnerable to rebinding between the
    validation and connection steps.
    """
    normalized = (host or "").rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _effective_port(url: httpx.URL) -> int:
    if url.port is not None:
        return int(url.port)
    return 443 if url.scheme == "https" else 80


def _validate_base_url(base_url: str) -> httpx.URL:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty absolute HTTP(S) URL")
    try:
        url = httpx.URL(base_url)
    except (httpx.InvalidURL, TypeError):
        # Configuration-controlled URL text may itself contain credentials.
        # Normalize the exception and do not chain HTTPX's detailed message.
        raise ValueError("base_url must be a valid absolute HTTP(S) URL") from None

    if url.scheme not in {"http", "https"} or not url.host:
        raise ValueError("base_url must be an absolute http:// or https:// URL")
    if url.username or url.password:
        raise ValueError("base_url must not contain userinfo; pass api_key= instead")
    if url.query:
        raise ValueError("base_url must not contain a query string")
    if url.fragment:
        raise ValueError("base_url must not contain a fragment")
    if url.path not in {"", "/"}:
        raise ValueError("base_url must contain only an origin, not a path prefix")
    if url.scheme == "http" and not _is_loopback_host(url.host):
        raise ValueError(
            "refusing plaintext HTTP to a non-loopback host; paths, queries, "
            "questions, and programs may be private — configure https:// explicitly"
        )
    return url


def _redirect_is_safe(source: httpx.URL, target: httpx.URL) -> bool:
    """Allow same-origin redirects and a tightly scoped same-host TLS upgrade."""
    if target.scheme not in {"http", "https"} or not target.host:
        return False
    if target.username or target.password:
        return False
    if source.host.lower() != target.host.lower():
        return False
    source_port = _effective_port(source)
    target_port = _effective_port(target)
    if source.scheme == target.scheme:
        return source_port == target_port
    if source.scheme == "http" and target.scheme == "https":
        return (
            source_port == target_port
            or (source_port == 80 and target_port == 443)
        )
    return False


def _seg(value: Any) -> str:
    """Percent-encode one path segment. Without this a ticker containing '/'
    (or '^', '#', '?') silently re-routes the request to a different path and
    the caller sees an indistinguishable 404."""
    encoded = _urlquote(str(value), safe="")
    # RFC 3986 leaves dots unescaped, but clients normalize exact '.' and '..'
    # segments before transmission and can escape the intended API route.
    if encoded in {".", ".."}:
        return encoded.replace(".", "%2E")
    return encoded


class ApiError(RuntimeError):
    """A request did not yield a usable JSON payload.

    Raised for every non-2xx response (`status_code` is the HTTP status and
    `detail` is the server's `detail` field when it sent one), and also for a
    2xx whose body could not be decoded as JSON — which in practice means a
    proxy or captive portal answered instead of the API.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = int(status_code)
        self.detail = detail


def _reject_malformed_redirect_location(response: httpx.Response) -> None:
    """Normalize malformed redirects before HTTPX builds a next request.

    HTTPX constructs ``response.next_request`` even with automatic redirect
    following disabled. An invalid Location can otherwise escape Client.send
    as a transport exception before the SDK's trust-boundary check sees it.
    The attacker-controlled header is deliberately omitted from the error.
    """
    if not response.has_redirect_location:
        return
    try:
        httpx.URL(response.headers["location"])
    except httpx.InvalidURL:
        response.close()
        raise ApiError(
            response.status_code, "invalid redirect Location header"
        ) from None


class AlelyonClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, *,
                 api_key: Optional[str] = None, timeout: float = 60.0,
                 follow_redirects: bool = True,
                 transport: Optional[httpx.BaseTransport] = None,
                 max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
                 max_redirects: int = DEFAULT_MAX_REDIRECTS) -> None:
        if api_key is not None and not isinstance(api_key, str):
            raise TypeError("api_key must be a string or None")
        if api_key is not None and (
            not MIN_API_KEY_LENGTH <= len(api_key) <= MAX_API_KEY_LENGTH
            or any(not 33 <= ord(character) <= 126 for character in api_key)
        ):
            raise ValueError(
                "api_key must be a 32-512 character visible ASCII bearer token"
            )
        if not isinstance(follow_redirects, bool):
            raise TypeError("follow_redirects must be a bool")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a finite positive number of seconds")
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite positive number of seconds")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes <= 0
        ):
            raise ValueError("max_response_bytes must be a positive integer")
        if (
            isinstance(max_redirects, bool)
            or not isinstance(max_redirects, int)
            or max_redirects < 0
        ):
            raise ValueError("max_redirects must be a non-negative integer")

        parsed_base = _validate_base_url(base_url)
        # A decoded-size check runs after HTTPX's content decoder. Asking for
        # identity and refusing a server that ignores it prevents a compressed
        # response from allocating its expanded body before our cap can run.
        headers = {"accept-encoding": "identity"}
        if api_key is not None:
            headers["authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(
            base_url=parsed_base,
            headers=headers,
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
            # Loopback HTTP must travel directly to loopback. Otherwise an
            # inherited HTTP_PROXY can silently disclose a local request (and,
            # when configured, its bearer key) to a plaintext proxy.
            trust_env=not _is_loopback_host(parsed_base.host),
            event_hooks={"response": [_reject_malformed_redirect_location]},
        )
        self._timeout_s = timeout
        self._follow_redirects = follow_redirects
        self._max_response_bytes = max_response_bytes
        self._max_redirects = max_redirects

    # ── plumbing ─────────────────────────────────────────────────────────────
    def _get(self, path: str, **params: Any) -> Any:
        request = self._http.build_request(
            "GET",
            path,
            params={k: v for k, v in params.items() if v is not None},
        )
        return self._unwrap(self._send(request))

    def _post(self, path: str, payload: dict) -> Any:
        request = self._http.build_request("POST", path, json=payload)
        return self._unwrap(self._send(request))

    def _send(self, request: httpx.Request) -> httpx.Response:
        """Send one bounded request, enforcing redirect trust before each hop."""
        deadline = time.monotonic() + self._timeout_s
        redirects = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise httpx.TimeoutException(
                    "Alelyon request exceeded its total timeout", request=request
                )
            phase_timeouts = request.extensions.get("timeout", {})
            request.extensions["timeout"] = {
                name: (
                    remaining
                    if value is None
                    else min(float(value), remaining)
                )
                for name, value in phase_timeouts.items()
            }

            response = self._http.send(
                request, stream=True, follow_redirects=False
            )
            if not response.has_redirect_location or not self._follow_redirects:
                return self._bounded_response(response, deadline)

            next_request = response.next_request
            if next_request is None or not _redirect_is_safe(
                response.request.url, next_request.url
            ):
                status = response.status_code
                response.close()
                raise ApiError(
                    status,
                    "refusing unsafe redirect before sending a second request",
                )

            redirects += 1
            if redirects > self._max_redirects:
                status = response.status_code
                response.close()
                raise ApiError(
                    status,
                    f"refusing more than {self._max_redirects} redirects",
                )
            response.close()
            request = next_request

    def _bounded_response(
        self, response: httpx.Response, deadline: float
    ) -> httpx.Response:
        """Read under a decoded-size cap and between-I/O deadline checkpoints."""
        if time.monotonic() >= deadline:
            response.close()
            raise httpx.TimeoutException(
                "Alelyon response exceeded its total timeout",
                request=response.request,
            )
        content_encoding = response.headers.get("content-encoding", "").strip().lower()
        if content_encoding not in {"", "identity"}:
            response.close()
            raise ApiError(
                response.status_code,
                "refusing encoded response; Alelyon requires Content-Encoding: identity",
            )
        declared = response.headers.get("content-length")
        if declared:
            try:
                if int(declared) > self._max_response_bytes:
                    response.close()
                    raise ApiError(
                        response.status_code,
                        "response exceeds max_response_bytes "
                        f"({self._max_response_bytes})",
                    )
            except ValueError:
                pass

        body = bytearray()
        try:
            for chunk in response.iter_bytes():
                if time.monotonic() >= deadline:
                    raise httpx.TimeoutException(
                        "Alelyon response exceeded its total timeout",
                        request=response.request,
                    )
                if len(body) + len(chunk) > self._max_response_bytes:
                    raise ApiError(
                        response.status_code,
                        "response exceeds max_response_bytes "
                        f"({self._max_response_bytes})",
                    )
                body.extend(chunk)
            if time.monotonic() >= deadline:
                raise httpx.TimeoutException(
                    "Alelyon response exceeded its total timeout",
                    request=response.request,
                )
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                content=bytes(body),
                request=response.request,
                extensions=response.extensions,
            )
        finally:
            response.close()

    @staticmethod
    def _unwrap(resp: httpx.Response) -> Any:
        if resp.status_code // 100 != 2:
            if not json_nesting_within_limit(resp.content):
                detail = "invalid or excessively nested error response"
            else:
                try:
                    decoded = resp.json(
                        parse_constant=reject_nonfinite_constant,
                        parse_float=parse_finite_float,
                        parse_int=parse_bounded_int,
                        object_pairs_hook=reject_duplicate_pairs,
                    )
                    detail = (
                        decoded.get("detail", resp.text)
                        if isinstance(decoded, dict)
                        else resp.text
                    )
                except Exception:  # noqa: BLE001
                    content_type = resp.headers.get("content-type", "")
                    media_type = content_type.partition(";")[0].strip().lower()
                    if media_type == "application/json" or media_type.endswith(
                        "+json"
                    ):
                        detail = "invalid JSON error response"
                    else:
                        detail = resp.text
            if not detail and resp.is_redirect:
                # Location may hold credentials or a presigned query. Never
                # repeat attacker-controlled redirect targets in exceptions.
                detail = (
                    "redirect response refused "
                    "(construct the client with follow_redirects=True)"
                )
            raise ApiError(resp.status_code, str(detail))
        if resp.status_code == 204:
            return None
        if not resp.content:
            raise ApiError(
                resp.status_code,
                "expected a JSON body, got an empty successful response",
            )
        if not json_nesting_within_limit(resp.content):
            raise ApiError(
                resp.status_code,
                "expected a JSON body within the maximum nesting depth",
            )
        try:
            return resp.json(
                parse_constant=reject_nonfinite_constant,
                parse_float=parse_finite_float,
                parse_int=parse_bounded_int,
                object_pairs_hook=reject_duplicate_pairs,
            )
        except Exception as exc:  # noqa: BLE001
            raise ApiError(
                resp.status_code,
                "expected a JSON body, got "
                f"{resp.headers.get('content-type', 'no content-type')} "
                f"({type(exc).__name__})",
            ) from exc

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "AlelyonClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── system ───────────────────────────────────────────────────────────────
    def health(self) -> dict:
        return self._get("/v1/system/health")

    def certificate(self, table: str, scope1: str, scope2: str) -> dict:
        return self._get(
            f"/v1/system/certificates/{_seg(table)}/{_seg(scope1)}/{_seg(scope2)}")

    # ── market ───────────────────────────────────────────────────────────────
    def bars(self, ticker: str, *, lookback_days: int = 400,
             interval: str = "1d", prefer_store: bool = False) -> dict:
        return self._get(f"/v1/market/bars/{_seg(ticker)}",
                         lookback_days=lookback_days, interval=interval,
                         prefer_store=prefer_store)

    def quote(self, ticker: str) -> dict:
        return self._get(f"/v1/market/quote/{_seg(ticker)}")

    def quotes(self, tickers: Iterable[str], *,
               include_certs: bool = False) -> dict:
        """Batched last prices. Pass include_certs=True for the per-symbol
        capture certificates (off by default — they roughly double the payload).
        Raises ValueError past MAX_QUOTE_SYMBOLS rather than let the server
        silently truncate the batch."""
        if isinstance(tickers, str):
            raise TypeError("tickers must be an iterable of symbols, not a string")
        syms = [str(t).strip().upper() for t in tickers if str(t).strip()]
        if len(syms) > MAX_QUOTE_SYMBOLS:
            raise ValueError(
                f"quotes() accepts at most {MAX_QUOTE_SYMBOLS} symbols per call "
                f"(got {len(syms)}) — the server truncates silently past that, "
                f"so chunk the request yourself")
        return self._get("/v1/market/quotes", tickers=",".join(syms),
                         include_certs=include_certs)

    # ── macro ────────────────────────────────────────────────────────────────
    def fred(self, series_id: str, *, lookback_days: int = 3650,
             prefer_store: bool = False) -> dict:
        return self._get(f"/v1/macro/series/{_seg(series_id)}",
                         lookback_days=lookback_days, prefer_store=prefer_store)

    def indicators(self) -> dict:
        return self._get("/v1/macro/indicators")

    def pulse(self) -> dict:
        return self._get("/v1/macro/pulse")

    # ── desks ────────────────────────────────────────────────────────────────
    def rates(self) -> dict:
        return self._get("/v1/desks/rates")

    def vol(self, ticker: str) -> dict:
        return self._get(f"/v1/desks/vol/{_seg(ticker)}")

    def rotation(self) -> dict:
        return self._get("/v1/desks/rotation")

    def crash(self) -> dict:
        return self._get("/v1/desks/crash")

    def breadth(self) -> dict:
        return self._get("/v1/desks/breadth")

    # ── intelligence ─────────────────────────────────────────────────────────
    def analyst(self, ticker: str) -> dict:
        return self._get(f"/v1/analyst/{_seg(ticker)}")

    def screener(self, symbols: Iterable[str] = (), *,
                 timeframe: str = "Daily") -> list:
        """The L/S screener board — a LIST of tiles, one per symbol. An empty
        `symbols` requests the server's default board. Unrecognized timeframes
        fall back to Daily server-side. The server admits at most 100 bounded
        market-symbol tokens and raises ApiError(422) for an invalid universe."""
        if isinstance(symbols, str):
            raise TypeError("symbols must be an iterable of symbols, not a string")
        return self._get("/v1/screener/board",
                         symbols=",".join(symbols), timeframe=timeframe)

    def answer(self, question: str) -> dict:
        return self._post("/v1/answer", {"question": question})

    # ── certified (signed, portable envelopes) ───────────────────────────────
    def certified_pubkey(self) -> dict:
        """The issuer's signing public key. PIN THIS out-of-band: verifying an
        envelope against a key carried inside that same envelope authenticates
        nothing."""
        return self._get("/v1/certified/pubkey")

    def certified_witness_pubkey(self) -> dict:
        """The co-signing witness's public key. Check the `independent` field —
        a witness co-located with the signer demonstrates the wire format but
        gives no equivocation resistance."""
        return self._get("/v1/certified/witness/pubkey")

    def certify(self, *, program: Optional[str] = None,
                question: Optional[str] = None,
                K: int = 63, alpha: float = 0.05) -> dict:
        """Issue a signed Certified Number Envelope for a DSL `program` (the
        deterministic path) or a natural-language `question` the server's LLM
        turns into one. Verify the reply yourself against a pinned public key.
        Raises ApiError 422 when neither yields a program, 503 when a `question`
        was given but no LLM author is configured."""
        if not (program or "").strip() and not (question or "").strip():
            raise ValueError("supply either program= or question=")
        payload: dict = {"K": int(K), "alpha": float(alpha)}
        if program:
            payload["program"] = program
        if question:
            payload["question"] = question
        return self._post("/v1/certified/envelope", payload)

    # ── engine (read-only) ───────────────────────────────────────────────────
    def engine_status(self) -> dict:
        return self._get("/v1/engine/status")

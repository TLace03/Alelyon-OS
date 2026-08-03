"""alelyon.platform.sdk — the client SDK for the Alelyon platform.

Two clients, neither of which imports anything server-side (safe on any machine
that can reach the API host / engine host):

- AlelyonClient  — HTTP client for the read-only v1 API: certified market data
  and its capture certificates, macro, the desk reads, analyst, screener,
  certified answers, signed envelopes, engine status.
- EngineStream   — ZMQ subscriber for a running engine's telemetry broadcast
  (health / pnl / position / signal / order / feature / tick / news).

Verifying a signed envelope needs no engine and no key: use `alelyon.verify`,
which ships in the same `alelyon-os` distribution as this client.

Other languages: the API serves /openapi.json, so any OpenAPI generator can
produce a typed client.
"""
from __future__ import annotations

from alelyon.platform.sdk.client import (AlelyonClient, ApiError,
                                         DEFAULT_BASE_URL, MAX_QUOTE_SYMBOLS)
from alelyon.platform.sdk.stream import EngineStream

__version__ = "0.1.0"

__all__ = ["AlelyonClient", "ApiError", "EngineStream", "DEFAULT_BASE_URL",
           "MAX_QUOTE_SYMBOLS", "__version__"]

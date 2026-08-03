"""`alelyon-verify` — the command-line face of the open verifier (Track 0, W1).

Three jobs, and the reason each exists:

  selftest   Run the bundled conformance suite. This is W1's acceptance test: on a
             machine that is not the dev box, `pip install alelyon-os` then
             `alelyon-verify selftest` must report every golden verifying and every
             forgery failing — with no repo checkout anywhere in the picture.

  verify     Verify one envelope against the caller's own data and a key they
             pinned out of band. This is what the external party runs at the
             first-external-verification milestone (PLATFORM.md §5). It prints a
             machine-readable verdict and exits nonzero when the envelope does not
             verify, so it composes into a script without anyone reading prose.

  version    Print what this build is and, critically, WHICH envelope types it
             claims to verify and which numeric substrate it is on. A verifier that
             cannot say what it supports cannot be pinned to a spec revision
             (W1(d)), and a width verified on an unspecified substrate is not
             portable (SPEC §8.1) — so both are printed every time.

The `verify` and `selftest` output shapes are the uniform protocol a
second-language implementation is driven through, so divergences show up as a
diff rather than as an argument (the sigstore-conformance pattern).

Data file format, identical to a vector's `inputs` block (SPEC §12):

    {"price|SYN": {"index": [1704153600.0, ...], "values": [100.1, ...]}}

Exit codes: 0 verified / suite clean · 1 not verified / suite failed ·
2 usage or I/O error. A skipped-but-not-failed suite exits 0 and says how many
were skipped; a suite that skipped everything is not a pass and exits 1.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Dict, List, Optional


_MAX_JSON_FILE_BYTES = 64 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 2_000_000
_MAX_JSON_CONTAINER_ITEMS = 1_000_000
_MAX_JSON_STRING_BYTES = 1024 * 1024
_MAX_JSON_INTEGER_DIGITS = 1024


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number is outside finite f64 range")
    return parsed


def _parse_bounded_int(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > _MAX_JSON_INTEGER_DIGITS:
        raise ValueError("JSON integer exceeds the resource limit")
    return int(value)


def _unique_object(pairs):
    if len(pairs) > _MAX_JSON_CONTAINER_ITEMS:
        raise ValueError("JSON object exceeds the item limit")
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON object member: {name}")
        result[name] = value
    return result


def _nesting_within_limit(text: str) -> bool:
    depth = 0
    in_string = False
    escaped = False
    for value in text:
        if in_string:
            if escaped:
                escaped = False
            elif value == "\\":
                escaped = True
            elif value == '"':
                in_string = False
            continue
        if value == '"':
            in_string = True
        elif value in ("[", "{"):
            depth += 1
            if depth > _MAX_JSON_DEPTH:
                return False
        elif value in ("]", "}"):
            depth = max(0, depth - 1)
    return True


def _validate_resources(root) -> None:
    nodes = 0
    stack = [root]
    while stack:
        value = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ValueError("JSON document exceeds the node limit")
        if isinstance(value, dict):
            if len(value) > _MAX_JSON_CONTAINER_ITEMS:
                raise ValueError("JSON object exceeds the item limit")
            for name, child in value.items():
                if len(name.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
                    raise ValueError("JSON object key exceeds the string limit")
                stack.append(child)
        elif isinstance(value, list):
            if len(value) > _MAX_JSON_CONTAINER_ITEMS:
                raise ValueError("JSON array exceeds the item limit")
            stack.extend(value)
        elif isinstance(value, str) and \
                len(value.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
            raise ValueError("JSON string exceeds the string limit")


def _load_json(path: str):
    with open(path, "rb") as fh:
        raw = fh.read(_MAX_JSON_FILE_BYTES + 1)
    if len(raw) > _MAX_JSON_FILE_BYTES:
        raise ValueError(
            f"JSON input exceeds {_MAX_JSON_FILE_BYTES} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("input is not valid UTF-8 JSON") from exc
    if not _nesting_within_limit(text):
        raise ValueError(f"JSON input exceeds {_MAX_JSON_DEPTH} nesting levels")
    try:
        parsed = json.loads(
            text, parse_constant=_reject_constant,
            parse_float=_parse_finite_float, parse_int=_parse_bounded_int,
            object_pairs_hook=_unique_object)
    except (OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError("input is not bounded valid UTF-8 JSON") from exc
    _validate_resources(parsed)
    return parsed


def _input_data(spec: dict) -> Dict:
    from alelyon.verify.conformance import series_from
    out: Dict = {}
    for k, s in (spec or {}).items():
        kind, _, key = str(k).partition("|")
        out[(kind, key)] = series_from(s)
    return out


def _cmd_version(args) -> int:
    from alelyon.runtime.oracle.dsl.envelope import _kernel_id
    from alelyon.verify import (SPEC_VERSION, SUPPORTED_ENVELOPE_TYPES,
                               __version__)
    kernel = _kernel_id()
    info = {
        "verifier_version": __version__,
        "spec_version": SPEC_VERSION,
        "supported_envelope_types": list(SUPPORTED_ENVELOPE_TYPES),
        "kernel": kernel,
        # Stated, not implied: only the native kernel has frozen numeric
        # semantics, so only on it can a width be verified portably.
        "substrate_specified": kernel.startswith("alelyon-vector/"),
    }
    print(json.dumps(info, indent=2, sort_keys=True))
    if not info["substrate_specified"]:
        print("note: this build is on the numpy fallback, whose reduction order is "
              "not frozen by the spec. A width verified here is not portable. "
              "No owner-approved public deterministic-kernel distribution is "
              "currently available; nonzero widths remain unverified here.",
              file=sys.stderr)
    return 0


def _cmd_selftest(args) -> int:
    from alelyon.verify.conformance import format_report, run_conformance
    report = run_conformance()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(format_report(report))
    if report["failed"]:
        return 1
    if report["total"] and report["passed"] == 0:
        print("every case was skipped — that is not a pass", file=sys.stderr)
        return 1
    return 0


def _cmd_vectors(args) -> int:
    from alelyon.verify.conformance import load_vectors
    rows = [{"slug": c.get("slug"), "must": c.get("must"),
             "description": c.get("description"),
             "kernel": (c.get("envelope") or {}).get("kernel")}
            for c in load_vectors()]
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def _cmd_manifest(args) -> int:
    """Check an issuer's published key history against a root pinned out of band.

    Separate from `verify` because it answers a different question, on a different
    cadence: `verify` asks about one number, this asks whether the key that signed it
    is still one you should accept. A client pins the root and checkpoint key, retains
    the last accepted signed checkpoint, and advances that state only after success.
    """
    from alelyon.runtime.atlas.data.keylife import (key_status_at,
                                                    verify_manifest_checkpoint)
    try:
        manifest = _load_json(args.manifest)
        checkpoint = _load_json(args.checkpoint)
        trusted_checkpoint = _load_json(args.trusted_checkpoint)
    except (OSError, ValueError) as exc:
        print(f"cannot read manifest/checkpoint: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2
    checkpointed = verify_manifest_checkpoint(
        manifest, checkpoint, root_public_key_hex=args.root,
        checkpoint_public_key_hex=args.checkpoint_key,
        trusted_checkpoint=trusted_checkpoint)
    v = checkpointed.get("manifest") or {"keys": {}, "order": []}
    out = {"ok": checkpointed["ok"], "reason": checkpointed["reason"],
           "failure": checkpointed.get("failure"), "chain": v["order"],
           "issuer": manifest.get("issuer") if isinstance(manifest, dict) else None}
    if checkpointed["ok"]:
        out["next_checkpoint"] = checkpointed["next_checkpoint"]
        out["keys"] = [
            {"key_id": k,
             "status": v["keys"][k].get("status"),
             "not_before": v["keys"][k].get("not_before"),
             "not_after": v["keys"][k].get("not_after"),
             "revocation": v["keys"][k].get("revocation")}
            for k in v["order"]]
        if args.at is not None:
            out["at"] = args.at
            out["status_at"] = {k: key_status_at(v, k, args.at)[0] for k in v["order"]}
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 0 if checkpointed["ok"] else 1


def _cmd_verify(args) -> int:
    from alelyon.verify import verify_envelope
    try:
        cne = _load_json(args.envelope)
        data = _input_data(_load_json(args.data)) if args.data else None
    except (OSError, ValueError) as exc:
        print(f"cannot read input: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.key is None:
        # Not a convenience default: a verification without a pinned key
        # authenticates nothing, so the tool refuses to look like it did
        # something. It still RUNS (the result is informative) but says so.
        print("warning: no --key pinned; the envelope's own embedded key cannot "
              "authenticate it, so ok=false is the only honest outcome",
              file=sys.stderr)

    try:
        manifest = _load_json(args.key_manifest) if args.key_manifest else None
        checkpoint = (_load_json(args.manifest_checkpoint)
                      if args.manifest_checkpoint else None)
        trusted_checkpoint = (_load_json(args.trusted_manifest_checkpoint)
                              if args.trusted_manifest_checkpoint else None)
    except (OSError, ValueError) as exc:
        print(f"cannot read manifest: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    result = verify_envelope(cne, data, public_key_hex=args.key,
                             witness_key_hex=args.witness_key,
                             key_manifest=manifest,
                             manifest_root_hex=args.manifest_root,
                             manifest_checkpoint=checkpoint,
                             checkpoint_public_key_hex=args.checkpoint_key,
                             trusted_manifest_checkpoint=trusted_checkpoint)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="alelyon-verify",
        description="Verify a Certified Number Envelope by replay against your own "
                    "copy of the inputs, under a key you pinned out of band.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("version", help="print version, spec, and substrate")
    sp.set_defaults(fn=_cmd_version)

    sp = sub.add_parser("selftest", help="run the bundled conformance suite")
    sp.add_argument("--json", action="store_true", help="machine-readable report")
    sp.set_defaults(fn=_cmd_selftest)

    sp = sub.add_parser("vectors", help="list the bundled conformance vectors")
    sp.set_defaults(fn=_cmd_vectors)

    sp = sub.add_parser("manifest", help="check an issuer's key history")
    sp.add_argument("--manifest", required=True, help="path to the key manifest JSON")
    sp.add_argument("--root", required=True,
                    help="the issuer's ROOT ed25519 public key, 64 hex "
                                  "chars, obtained OUT OF BAND. Required: a chain "
                                  "checked against nothing vouches for nothing")
    sp.add_argument("--checkpoint", required=True,
                    help="path to the signed checkpoint for this manifest")
    sp.add_argument("--checkpoint-key", required=True,
                    help="checkpoint ed25519 public key obtained OUT OF BAND")
    sp.add_argument("--trusted-checkpoint", required=True,
                    help="previous signed checkpoint retained by this verifier, or "
                         "the initial checkpoint obtained out of band")
    sp.add_argument("--at", type=float, help="epoch seconds; report each key's "
                                             "status at that moment")
    sp.set_defaults(fn=_cmd_manifest)

    sp = sub.add_parser("verify", help="verify one envelope")
    sp.add_argument("--envelope", required=True, help="path to the CNE JSON")
    sp.add_argument("--data", help="path to your copy of the inputs (see --help)")
    sp.add_argument("--key", help="signer's ed25519 public key, 64 hex chars, "
                                  "obtained OUT OF BAND")
    sp.add_argument("--witness-key", help="witness's ed25519 public key, 64 hex "
                                          "chars, obtained out of band")
    sp.add_argument("--key-manifest", help="path to the issuer's key manifest; "
                                          "reports whether the signing key was in "
                                          "service, and refuses a revoked one")
    sp.add_argument("--manifest-root", help="the ROOT key the manifest chains to, "
                                           "obtained out of band (required with "
                                           "--key-manifest)")
    sp.add_argument("--manifest-checkpoint",
                    help="signed checkpoint committing --key-manifest")
    sp.add_argument("--checkpoint-key",
                    help="checkpoint public key obtained out of band")
    sp.add_argument("--trusted-manifest-checkpoint",
                    help="previous signed checkpoint retained by this verifier")
    sp.set_defaults(fn=_cmd_verify)
    return p


def _configure_stdio() -> None:
    """Never let ENCODING be the reason a verification fails.

    Found by the clean-install acceptance test, which is the point of having one:
    `alelyon-verify selftest` crashed with UnicodeEncodeError on a stock Windows
    console, because the report says "Δ" and the console codepage was cp1252. That
    is a nonzero exit from the first command an external party runs, on a verdict
    that was actually clean — the worst possible failure mode for a tool whose exit
    code is the answer.

    `backslashreplace` rather than `replace`: an unrenderable character degrades to
    an escape that still says which character it was, instead of to `?`.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError, OSError):   # pragma: no cover
            pass                                        # not a reconfigurable stream


def main(argv: Optional[List[str]] = None) -> int:
    _configure_stdio()
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":       # pragma: no cover - process entry point
    raise SystemExit(main())

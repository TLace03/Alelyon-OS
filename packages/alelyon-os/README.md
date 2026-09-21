# alelyon-os

This distribution contains Alelyon's reviewed public Python surface. The exact
module set is declared in `subsystems.py` and checked by `build_wheel.py`.
The private monorepo contains additional products and runtime components that
are not part of this package.

## Install

```sh
python -m pip install alelyon-os
alelyon-verify selftest
```

The package manifest requires Python 3.10 or newer. The source version in
`pyproject.toml` is a build input, not proof that version is on an index. Read
the installed version and conformance result for the artifact you use.

This wheel is not minimal: every installation receives the full reviewed file
set. Extras gate dependencies, not files.

The API is unstable and carries no compatibility promise before 1.0. Pin the
exact artifact and test upgrades against the operations you depend on.

| Extra | Dependencies from the package manifest |
|---|---|
| `sdk` | `httpx>=0.27` |
| `stream` | `pyzmq>=25` |
| `dev` | `pytest>=7`, `httpx>=0.27` |

## Public surfaces

The allowlist includes CNE production/replay, coordinate registration and model
metadata morphometry, uncertainty-aware compute, fleet/worktree coordination,
the workspace conversation surface and the API client. Consult the actual
allowlist before assuming a private-source API is available in an install.
The private capture/history services, desktop, broker workflows and identity
infrastructure do not become public because they share a namespace.

The package installs these entry points:

| Command | Implementation |
|---|---|
| `alelyon-verify` | `alelyon.verify.cli:main` |
| `alelyon-fleet` | `alelyon.runtime.common.fleet_cli:main` |
| `alelyon-chat` | `alelyon.runtime.common.chat_cli:main` |
| `alelyon-ledger` | `alelyon.runtime.common.fleet_ledger_cli:main` |
| `alelyon-workspace` | `alelyon.runtime.oracle.assistant.cli:main` |

## Verify a receipt

```sh
alelyon-verify verify --envelope receipt.json --data inputs.json --key PINNED_PUBLIC_KEY_HEX
```

Obtain the issuer's public key through a trusted channel independent of the
receipt. Supply your own input extract. A key embedded in the received receipt
does not authenticate its issuer. A passing replay checks commitments and the
computed result; it does not establish that the inputs were true at capture.

Substrate-sensitive nonzero widths need the specified deterministic kernel for
full replay. The fallback leaves that width unverified. Exact-zero widths can
fully replay without that requirement. Record the complete verdict and reasons;
successful JSON parsing or HTTP transport is not verification success.

## Development and release boundary

The public mirror is generated upstream. Edit the source allowlist and code,
not generated mirror files. Maintainers validate staging, closure and clean
installation with the dedicated builder; public export and package publication
are separate authorized actions. Source traceability manifests describe an
origin but are not authenticated proof of who ran the export.

```sh
python packaging/alelyon-os/build_wheel.py --check-only
python packaging/alelyon-os/build_wheel.py --check-closure
python packaging/alelyon-os/build_wheel.py --verify-clean-install
```

These are source-maintainer commands, run from the private repository root.
They do not substitute for the native, conformance and publication gates for
the exact release.

## Migrating from the old packages

Use `alelyon-os` as the distribution. The verifier command remains
`alelyon-verify`; the SDK import is `alelyon.platform.sdk`. The package's
migration contract marks the former distribution names `alelyon-sdk`,
`alelyon-verify` and `alelyon-mock` as no longer registered to Alelyon. Do not
install from those names or assume that a later index listing belongs to this
project.

## License

The package manifest specifies `Apache-2.0 OR MIT`, at your option. The builder
includes both `LICENSE-APACHE` and `LICENSE-MIT`; select either license under
its terms.

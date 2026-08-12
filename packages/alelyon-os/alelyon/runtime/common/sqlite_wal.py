"""WAL mode and SQLITE_BUSY retry, for every store in `runtime.common`.

Extracted from `db.py` rather than imported from it, and the reason is a
shipping boundary rather than tidiness. `db.py` is `fam.db`'s module — the
markets database, its schema, its legacy data directory. The fleet stores need
exactly two things from it (`try_set_wal`, `with_busy_retry`), and the fleet
subsystem is PUBLISHED in `alelyon-os`, so importing `db` from `fleet_ledger`
made a fresh install of that distribution raise ImportError unless the whole of
`fam.db`'s module travelled with it. Publishing the markets foundation to obtain
a WAL pragma is the wrong trade in both directions: it widens a public surface
with content that has nothing to do with worktree coordination, and it does not
make the coordination code any clearer.

So the primitives live here, `db.py` imports them, and the fleet stores import
them. One implementation, and the published subsystem carries ~90 lines of
generic SQLite plumbing instead of a database it does not use.

Stdlib only, and that is a SHIPPING CONSTRAINT rather than an accident: every
module in the fleet file list must be importable from `pip install alelyon-os`
with nothing further named, because this subsystem declares no extra of its own.

Deliberately phrased without a bracketed extra after the distribution name.
`test_every_extra_named_in_shipped_code_is_declared` scans every shipped file
for one and requires it to be declared, and this subsystem declares none: it
needs no third-party package, and pyproject's doctrine is that an extra
resolving to nothing is worse than no extra, because the install succeeds, does
nothing further, and warns nobody. The scan matches the literal text ANYWHERE in
the file, so a sentence cautioning against writing one still trips it -- which is
why this paragraph describes the form instead of quoting it.
"""
from __future__ import annotations

import sqlite3
import time

#: Substrings SQLite uses for SQLITE_BUSY / SQLITE_LOCKED. Matching on the
#: message is unpleasant and is what the stdlib driver leaves available:
#: `sqlite3.OperationalError` carries no errno on Python 3.10, and
#: `sqlite3.Error.sqlite_errorcode` (3.11+) is checked first where present.
_BUSY_TEXT = ("database is locked", "database table is locked",
              "database schema is locked")

#: SQLITE_BUSY / SQLITE_LOCKED primary result codes.
_BUSY_CODES = frozenset({5, 6})


def set_wal(c: sqlite3.Connection, attempts: int = 6) -> None:
    """Put the connection in WAL mode, tolerating a concurrent first-touch.

    `PRAGMA journal_mode=WAL` needs a brief exclusive lock and — unlike ordinary
    statements — does NOT honour busy_timeout: it returns SQLITE_BUSY at once.
    So when several threads open a COLD database simultaneously (the db already
    being in WAL makes the pragma a no-op, which is why this only bites on a
    fresh file), all but one used to fail with "database is locked" and the
    whole connection would fail to open.

    Retry briefly, then check whether someone else already made it WAL — if so
    we have what we wanted. Only a genuine failure to reach WAL raises, because
    silently running in rollback-journal mode would break the concurrent
    reader/writer guarantee the rest of `db` depends on.
    """
    for i in range(attempts):
        try:
            row = c.execute("PRAGMA journal_mode=WAL").fetchone()
            if row and str(row[0]).lower() == "wal":
                return
        except sqlite3.OperationalError:
            pass
        try:
            row = c.execute("PRAGMA journal_mode").fetchone()
            if row and str(row[0]).lower() == "wal":
                return                      # another connection got there first
        except sqlite3.OperationalError:
            pass
        time.sleep(0.02 * (i + 1))
    # Last attempt, unguarded: if it still fails the caller should see why.
    c.execute("PRAGMA journal_mode=WAL")


def try_set_wal(conn: sqlite3.Connection, attempts: int = 6) -> bool:
    """`set_wal` where WAL is an improvement rather than a guarantee.

    The fleet stores want WAL because a rollback journal makes every writer take
    an EXCLUSIVE whole-database lock and blocks every reader — measured on this
    workstation 2026-08-11: five live fleet stores, `journal_mode=delete` on all
    five, the largest 91 MB. What they must NOT do is refuse to open because a
    filesystem cannot do WAL. A network share, a read-only mount or a container
    overlay can all fail the pragma, and "the fleet bus will not open here" is a
    worse outcome than "the fleet bus is slower here".

    Returns whether WAL was reached, so a caller that wants to report the real
    journal mode can. `db.py`'s own connections keep using `set_wal`, which
    raises, because `fam.db`'s concurrent reader/writer guarantee does depend on
    it.
    """
    try:
        set_wal(conn, attempts)
        return True
    except sqlite3.Error:
        return False


def is_busy(exc: BaseException) -> bool:
    """True when `exc` is SQLITE_BUSY/SQLITE_LOCKED rather than a real fault.

    A busy error means "someone else holds the lock, ask again"; every other
    `OperationalError` means something is wrong and must not be retried into a
    spin. Telling them apart is the whole reason this is a named predicate and
    not an inline `except`.
    """
    if not isinstance(exc, sqlite3.Error):
        return False
    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int) and (code & 0xFF) in _BUSY_CODES:
        return True
    text = str(exc).lower()
    return any(fragment in text for fragment in _BUSY_TEXT)


def with_busy_retry(operation, *, attempts: int = 5, base_delay: float = 0.05,
                    max_delay: float = 0.8):
    """Run `operation()`, retrying ONLY on SQLITE_BUSY. Bounded, not a spin.

    `attempts` is a hard count and the backoff is capped, so the worst case is a
    stated number of sleeps and then the original error — never an unbounded
    wait and never a busy loop. A non-busy error propagates on the first raise,
    because retrying a schema error or a corrupt page just delays the report.

    `operation` must be safe to run again from the top. Every caller wraps a
    whole connect/transaction block, so a failed attempt has been rolled back
    before the next one starts.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    last: BaseException | None = None
    for index in range(attempts):
        try:
            return operation()
        except sqlite3.Error as exc:
            if not is_busy(exc):
                raise
            last = exc
            if index == attempts - 1:
                break
            time.sleep(min(max_delay, base_delay * (2 ** index)))
    assert last is not None
    raise last


__all__ = ["set_wal", "try_set_wal", "is_busy", "with_busy_retry"]

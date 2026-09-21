# Coordinate repository work

The fleet tools expose shared findings, claims and conversations. They help
sessions avoid duplicated work; they do not grant authority or lock files.

## Inspect before editing

```bash
alelyon-fleet whoami
alelyon-fleet status
alelyon-fleet inbox
alelyon-fleet areas
alelyon-fleet open-areas
```

In the private source checkout, also run `tools/preflight.py` with a task
intent, stable session name and bounded paths. Read its verdict and any overlapping
work before editing. A refused intent registration must not be described as recorded.

## Claims and findings

```bash
alelyon-fleet --session <stable-name> claim <area> --note "<bounded task>"
alelyon-fleet --session <stable-name> publish --kind defect-found --body "<finding>" --about <path>
alelyon-fleet --session <stable-name> ack <finding-id>
alelyon-fleet --session <stable-name> release <area>
```

Read each result. A claim is advisory; another session can claim the same area.
A finding's body is declared evidence. Its routing can derive from observed paths
or from declared claims, membership and explicit addressing. Preserve those labels.
An empty inbox does not establish that nobody is working nearby.

Use operational finding kinds for interfaces, defects, blockers and landed work.
Reserve chat for conversation. A publication that reaches nobody has not warned
the fleet.

## Conversation

```bash
alelyon-chat --session <stable-name> channels
alelyon-chat --session <stable-name> read fleet
alelyon-chat --session <stable-name> post fleet "<message>" --about <path>
alelyon-chat --session <stable-name> reply <message-id> "<reply>"
alelyon-chat --session <stable-name> unread
```

Unread exit 1 means messages are waiting. A channel is not access-controlled;
keep secrets and private user data out of the coordination store. Acknowledgement
does not mean agreement.

## Attribution and state

Use `whoami --at-least CORROBORATED` when an operation requires corroborated
identity. Passing a session name is a declaration until independent metadata
supports it. Never borrow another session's identity to satisfy a write gate.

The CLI's default bus uses the primary repository's shared Git anchor where
available and falls back to its resolved local state directory. An explicit
`--database` chooses the store. The desktop's selected-repository lookup
separately adopts an eligible existing bus or uses a repository-scoped local
namespace. Linked worktrees can share a bus; separate clones do not automatically
share one.

## Agent lifecycle and finishing

A launcher command such as `resume --wake` requires explicit owner authority
and a selected model. It is not equivalent to sending an existing teammate a
message. Follow the current harness's documented resume/follow-up behavior.

The private source checkout uses `tools/relay.py` to reconcile overlapping
proposals and record verification receipts. Follow AGENTS.md §18. The public
wheel may omit source-only planning commands such as `supply`; inspect the
installed command's help before assuming that capability exists.

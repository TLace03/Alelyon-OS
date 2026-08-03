#!/usr/bin/env bash
# Print, before the upload is attempted, WHICH trusted-publisher registration has to
# authorize it and on WHICH index — then the exact form values it must carry.
#
# Copy to Alelyon-OS as `.github/publisher-preflight.sh`.
#
# Why this exists
# ---------------
# `invalid-publisher` is the usual trusted-publishing failure and its diagnostic is
# awkward in a specific way: the publish action prints the OIDC *claims*, which are
# almost always correct, and says nothing about the two things that are actually
# wrong most of the time —
#
#   1. WHICH INDEX the form lives on. `pypi.org` and `test.pypi.org` are separate
#      services with separate accounts, separate logins and separate publisher
#      lists. A registration on one does nothing for an upload to the other, and the
#      rejection is byte-identical either way.
#
#   2. WHICH KIND of registration applies. A PENDING publisher (account level)
#      authorizes creating a project that does not exist yet. Once the project
#      exists, it is matched ONLY against publishers on its own settings page, and
#      the pending one stops applying. Which one governs depends on a fact about the
#      index — not on anything visible in the claims.
#
# So this reads the index (no credentials — it is a public endpoint) and states both.
# `docs/cne/PLATFORM.md` records four consecutive failed releases whose sole cause
# was a publisher registered against a repository that never ran the workflow; the
# claim table was correct through every one of them.
#
# Requests no OIDC token and prints no secret. Every value here is public.
set -euo pipefail

DIST="alelyon-os"
: "${INDEX_HOST:?INDEX_HOST must be set (pypi.org or test.pypi.org)}"
: "${TARGET_ENVIRONMENT:?TARGET_ENVIRONMENT must be set}"

owner="${GITHUB_REPOSITORY%%/*}"
repo="${GITHUB_REPOSITORY##*/}"

echo "Uploading to : https://${INDEX_HOST}/legacy/"
echo

# `|| true`: a preflight that fails the job on a flaky HEAD request would turn a
# diagnostic into a new source of red. An unreachable index is reported, not fatal.
code="$(curl -s -o /dev/null -w '%{http_code}' \
        "https://${INDEX_HOST}/pypi/${DIST}/json" || true)"

case "$code" in
  404)
    echo "${DIST} does NOT exist on ${INDEX_HOST} (HTTP 404)."
    echo
    echo "  => a PENDING publisher must authorize this upload, registered at"
    echo "     https://${INDEX_HOST}/manage/account/publishing/"
    echo
    echo "     A project-level registration cannot exist yet — there is no project"
    echo "     to hang it on. The pending publisher creates it on first success."
    ;;
  200)
    echo "${DIST} EXISTS on ${INDEX_HOST} (HTTP 200)."
    echo
    echo "  => this upload is matched ONLY against publishers on the project's own"
    echo "     settings page:"
    echo "     https://${INDEX_HOST}/manage/project/${DIST}/settings/publishing/"
    echo
    echo "     A pending publisher no longer applies to it, even if one is still"
    echo "     listed on the account page."
    ;;
  *)
    echo "Could not read https://${INDEX_HOST}/pypi/${DIST}/json (HTTP ${code:-none})."
    echo "Which registration governs is therefore UNKNOWN — check both pages by hand."
    ;;
esac

cat <<EOF

The form on ${INDEX_HOST} — not on any other index — must read exactly:

  PyPI Project Name : ${DIST}
  Owner             : ${owner}
  Repository name   : ${repo}
  Workflow name     : release.yml
  Environment       : ${TARGET_ENVIRONMENT}

Notes on the two fields that are most often subtly wrong:

  * Workflow name is the FILENAME, not the \`name:\` inside the file. It is
    \`release.yml\`, never \`Release\`.
  * Environment must be \`${TARGET_ENVIRONMENT}\` or BLANK. Blank matches any
    environment; a value must match exactly. This job asserts
    \`${TARGET_ENVIRONMENT}\`, so a form reading any other non-empty value is
    rejected — including the other index's environment name.

If the publish step still fails after this, compare the claim table it prints
against the five values above, field by field.
EOF

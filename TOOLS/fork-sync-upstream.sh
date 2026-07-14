#!/bin/sh
# Merge upstream mpv master into this fork's integration branch ("custom").
#
# This fork carries local features as commits on top of upstream mpv. Run
# this script from anywhere inside the checkout to pull in upstream changes;
# the feature commits are preserved (merge, not rebase, so no force-push is
# ever needed and the branch history stays stable).
#
# On conflict: git lists the conflicting files; resolve them, then
#     git add -A && git merge --continue && git push origin custom
#
# After a successful merge, rebuild and smoke-test before relying on the
# result: upstream refactors can change behavior around the local features
# without producing a textual conflict.
set -eu

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

git remote get-url upstream >/dev/null 2>&1 || \
    git remote add upstream https://github.com/mpv-player/mpv.git

git fetch upstream
git checkout custom
git merge --no-edit upstream/master
git push origin custom

echo "==> custom is up to date with upstream/master. Rebuild and test."

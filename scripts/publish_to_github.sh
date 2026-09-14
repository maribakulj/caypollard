#!/usr/bin/env bash
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required: https://cli.github.com/" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 1
fi

REPO_NAME="${1:-caypollard}"
VISIBILITY="${2:-private}"

case "$VISIBILITY" in
  public|private|internal) ;;
  *)
    echo "Visibility must be one of: public, private, internal" >&2
    exit 1
    ;;
esac

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated. Run: gh auth login" >&2
  exit 1
fi

if [ ! -d .git ]; then
  git init -b main
fi

if ! git config user.name >/dev/null 2>&1; then
  GH_LOGIN="$(gh api user --jq .login)"
  git config user.name "$GH_LOGIN"
fi

if ! git config user.email >/dev/null 2>&1; then
  GH_LOGIN="$(gh api user --jq .login)"
  git config user.email "${GH_LOGIN}@users.noreply.github.com"
fi

if ! git rev-parse HEAD >/dev/null 2>&1; then
  git add .
  git commit -m "chore: initialize research roadmap and reproducible project skeleton"
fi

gh repo create "$REPO_NAME" "--$VISIBILITY" \
  --description "Reproducible research on visual, knowledge-graph, and multimodal embeddings for cultural heritage retrieval." \
  --source=. \
  --remote=origin \
  --push

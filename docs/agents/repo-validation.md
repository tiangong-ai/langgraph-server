---
title: ai-langgraph-server Validation Guide
docType: guide
scope: repo
status: active
authoritative: false
owner: ai-langgraph-server
language: en
whenToUse:
  - when selecting local validation commands
  - when recording proof for documentation-governance changes
whenToUpdate:
  - when validation commands or local gate behavior changes
checkPaths:
  - docs/agents/repo-validation.md
  - .docpact/config.yaml
  - .githooks/pre-push
  - scripts/docpact
  - scripts/docpact-gate.sh
  - scripts/install-git-hooks.sh
lastReviewedAt: 2026-05-28
lastReviewedCommit: de6d888d8738e7c99b8fd142a5a8c36ca2f67db8
related:
  - ../../AGENTS.md
  - ../../.docpact/config.yaml
---

# ai-langgraph-server Validation Guide

## Local Docpact Push Gate

Install the versioned local hook once per checkout:

```bash
./scripts/install-git-hooks.sh
```

The `pre-push` hook runs `scripts/docpact-gate.sh`, which delegates CLI lookup to `scripts/docpact` and performs strict config validation plus enforced lint before the push leaves the machine. The wrapper checks `DOCPACT_BIN`, Cargo install locations, Homebrew install locations, and then `PATH`, so local agent shells should not fail only because bare `docpact` is unavailable. The default comparison base is `origin/main`. Override it for unusual stacks with `DOCPACT_BASE_REF=<ref>` or `scripts/docpact-gate.sh --base <ref>`. The gate writes its detailed report to a temporary file so normal pushes do not create `.docpact/runs/` artifacts.

## AgentPond Tracing

For tracing changes, run:

```bash
npx tsc --noEmit
eval "$(npx agentpond env get local)"
# Invoke one of the graphs declared in langgraph.json.
npx agentpond sync
npx agentpond traces list --limit 10
```

The validation is complete only when the graph invocation succeeds and the synced trace includes its LangGraph and LLM observations.

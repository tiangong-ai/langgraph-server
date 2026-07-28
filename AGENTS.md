---
title: ai-langgraph-server AI Working Guide
docType: contract
scope: repo
status: active
authoritative: true
owner: ai-langgraph-server
language: en
whenToUse:
  - when routing work from the workspace root into this repository
  - when deciding validation, ownership, or documentation-governance rules
  - when changing runtime, operator, or local gate behavior
whenToUpdate:
  - when repo ownership or runtime boundaries change
  - when validation or local gate behavior changes
  - when docpact governance rules change
checkPaths:
  - AGENTS.md
  - README.md
  - .docpact/config.yaml
  - docs/agents/**
  - .github/workflows/ai-doc-lint.yml
  - .githooks/**
  - scripts/docpact
  - scripts/docpact-gate.sh
  - scripts/install-git-hooks.sh
lastReviewedAt: 2026-05-28
lastReviewedCommit: de6d888d8738e7c99b8fd142a5a8c36ca2f67db8
related:
  - .docpact/config.yaml
  - docs/agents/repo-validation.md
---

# ai-langgraph-server AI Working Guide

LangGraph server runtime, agent orchestration, gateway, and local development repository.

## Local Docpact Push Gate

Install the versioned local hook once per checkout:

```bash
./scripts/install-git-hooks.sh
```

The `pre-push` hook runs `scripts/docpact-gate.sh`, which delegates CLI lookup to `scripts/docpact` and performs strict config validation plus enforced lint before the push leaves the machine. The wrapper checks `DOCPACT_BIN`, Cargo install locations, Homebrew install locations, and then `PATH`, so local agent shells should not fail only because bare `docpact` is unavailable. The default comparison base is `origin/main`. Override it for unusual stacks with `DOCPACT_BASE_REF=<ref>` or `scripts/docpact-gate.sh --base <ref>`. The gate writes its detailed report to a temporary file so normal pushes do not create `.docpact/runs/` artifacts.

## AgentPond Tracing

AgentPond tracing is opt-in. Keep `src/instrumentation.ts` inactive unless a Files SDK environment has set `FILES_SDK_PROVIDER`, and import it from every graph entry declared in `langgraph.json`. Validate tracing changes with `npx tsc --noEmit` and a real graph invocation followed by `agentpond sync`.

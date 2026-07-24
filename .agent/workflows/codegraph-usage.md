---
description: how to use the CodeGraph intelligence engine
---

# CodeGraph Intelligence Engine Usage Guide

CodeGraph builds a local graph database of your codebase, enabling instant structural traversal, dependency tracing, and semantic search.

> [!IMPORTANT]
> **CREDIT-SAVING GUIDELINE FOR AI AGENTS (Claude, Antigravity, Cursor, etc.)**:
> DO NOT recursively view files or call search tools in a loop to trace imports, find callers, or map code paths. This will quickly drain your token limits and API credits.
> Instead, run **local `codegraph` commands** (like `codegraph trace` or `codegraph dependents`) to traverse the entire dependency graph in a single local execution.

---

## 🏗️ Architecture & Boundaries (Services)

### What is a "Service"?
A service is a logical architectural boundary or module in your repository (e.g. backend, frontend, microservices). Boundaries are defined in `.codegraph/architecture.json`.
- **Per-Project Isolation**: CodeGraph databases are created **locally** per-project in `.codegraph/intelligence.db` (never globally).
- **Watcher (`codegraph watch`)**: The watcher monitors the project workspace and updates `intelligence.db` incrementally in real-time.

---

## 🛠️ CLI Reference

### 1. Scaffolding & Setup
- `codegraph init` — Scaffold workspace setup and default `architecture.json`.
- `codegraph build` — Perform full codebase parse and build the local SQLite database.

### 2. Analysis & Navigation (Credit-Saving Commands)
- `codegraph trace <from> <to>` — Traverses undirected relations to find the shortest structural path between two symbols (e.g., how a controller connects to a Prisma model).
- `codegraph dependents <symbol>` — Instantly list all classes, files, or methods that depend on or call a given symbol.
- `codegraph search "<query>"` — Quick FTS5 full-text search across symbols.
- `codegraph entrypoints` — Lists files/routes that are not imported by any other files.
- `codegraph doctor` — Diagnostic tool to check coverage and parser health.

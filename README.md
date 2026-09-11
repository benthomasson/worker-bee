# worker-bee 🐝

Belief-driven orchestrator for small-context worker bees.

Worker-bee reads a [reasons](https://github.com/benthomasson/ftl-reasons) belief database, identifies actionable issues (gated beliefs, contradictions, staleness), and dispatches focused tasks to LLMs via [Ollama](https://ollama.com/), [OpenAI](https://platform.openai.com/), or [Anthropic](https://www.anthropic.com/).

Each stage runs in a single context window (default budget: 56K tokens, targeting 64K context models). The orchestrator is plain Python control flow — the LLM is the worker bee inside each stage.

## Pipeline

```
reasons.db → Extract → Review → Trace → Verify → Fix → reasons.db
```

- **Extract** — queries reasons.db for gated beliefs, contradictions, stale beliefs, and unreviewed derivations
- **Review** — evaluates beliefs against their source documents in batches, stamps reviewed_at metadata, optionally retracts inaccurate beliefs via ftl-reasons TMS
- **Trace** — pure Python data gathering: walks belief → source summary → referenced code files, loads source within token budget
- **Verify** — dispatches LLM prompt with traced code to check whether source code supports each belief claim
- **Fix** — chains trace → verify → multi-turn code-editing loop with tool use
- **Edit** — standalone multi-turn code-editing loop (no belief context needed)

## Install

```
uv pip install -e .
```

## Usage

### Extract issues from a belief database

```
worker-bee extract path/to/reasons.db
```

### Review beliefs against source documents

```
worker-bee review path/to/reasons.db --model ollama:qwen-64k:latest
```

### Trace a belief back to source code (no LLM needed)

```
worker-bee trace path/to/reasons.db belief-id
```

### Verify beliefs against actual source code

```
worker-bee verify path/to/reasons.db                    # all unverified gated beliefs
worker-bee verify path/to/reasons.db belief-id          # single belief
```

### Fix a verified issue via code-editing loop

```
worker-bee fix path/to/reasons.db belief-id \
  --model ollama:qwen-64k:latest \
  --num-ctx 65536 \
  --verbose
```

### Standalone code editing (no belief database needed)

```
worker-bee edit "add error handling to parse_config()" \
  --model ollama:qwen-64k:latest \
  --num-ctx 65536
```

### Freeform prompt with tool use

Run a task with the full tool suite. The bee writes a summary entry when done.

```
worker-bee prompt "investigate the retry logic in dispatcher.py" \
  --db path/to/reasons.db \
  --model openai:gpt-4o
```

### Interactive chat

Each prompt gets a fresh context window. Knowledge persists across rounds via notes (`.worker-bee/notes.jsonl`) and beliefs.

```
worker-bee chat --model openai:gpt-4o --db path/to/reasons.db
worker-bee chat --model ollama:qwen-64k:latest --brain local.db --db hive.db
```

### Summarize a session log

Post-mortem summarization of a session's JSONL log into a dated entry.

```
worker-bee summarize .worker-bee/logs/20260910-233217.jsonl \
  --model ollama:qwen-64k:latest
```

### Run the full pipeline

```
worker-bee run path/to/reasons.db --model ollama:qwen-64k:latest
```

### Preview prompts without dispatching

```
worker-bee review path/to/reasons.db --dry-run
worker-bee fix path/to/reasons.db belief-id --dry-run
```

## Models

Worker-bee supports multiple LLM backends via model string prefixes:

| Prefix | Backend | Example |
|--------|---------|---------|
| `ollama:` | Local Ollama | `ollama:qwen-64k:latest`, `ollama:qwen3.8:27b` |
| `openai:` | OpenAI Responses API | `openai:gpt-4o`, `openai:gpt-4o-mini` |
| `api:` | Anthropic API (direct) | `api:claude-sonnet-4-20250514` |
| `claude` | Claude CLI | `claude`, `claude:haiku` |
| `gemini` | Gemini CLI | `gemini`, `gemini:gemini-2.5-flash` |

The `openai:` provider uses the Responses API (not Chat Completions) to support reasoning models with tool use.

### Flags

| Flag | Commands | Description |
|------|----------|-------------|
| `--model` | all | Model string (see table above) |
| `--dry-run` | all | Show prompts/tool calls without executing |
| `--verbose` | all | Print full prompts, tool calls, and token counts |
| `--limit` | review, verify | Max beliefs to process |
| `--batch-size` | review | Beliefs per LLM call (default: 5) |
| `--retract` | review | Retract beliefs found inaccurate (TMS cascade) |
| `--confirm` | fix, edit, prompt, chat | Y/N prompt before each tool call |
| `--max-turns` | fix, edit, prompt, chat | Max conversation turns (default: 20) |
| `--num-ctx` | fix, edit, prompt, chat | Ollama context window size (enables context tracking) |
| `--ctx-limit-pct` | fix, edit, prompt, chat | Stop at this fraction of context window (default: 0.80) |
| `--db` | edit, prompt, chat | Path to reasons.db (enables belief query tools) |
| `--brain` | fix, edit, prompt, chat | Path to local brain reasons.db (read/write, layered over --db) |
| `--truncate-chars` | fix, edit, prompt, chat | Truncate assistant output to N chars on stderr |
| `--no-questions` | chat | Disable the ask_user_question tool |

## Context Window Management

The task prompt is placed in the system message so Ollama preserves it when evicting old messages. Context usage is tracked each turn and the session stops at 80% capacity to prevent overflow.

Memory tools (`list_memory`, `retrieve_memory`, `write_note`) let the model page through evicted tool call results and keep scratch notes that survive context eviction.

Persistent notes (`.worker-bee/notes.jsonl`) survive across sessions. The bee reads them at the start of each task via `list_notes` and records findings as it works.

## Requirements

- Python 3.10+
- At least one LLM backend:
  - [Ollama](https://ollama.com/) with a 64K+ context model (e.g. Qwen 3 27B)
  - [OpenAI API key](https://platform.openai.com/) (`pip install openai`)
  - [Anthropic API key](https://console.anthropic.com/) (`pip install anthropic`)
- A [reasons.db](https://github.com/benthomasson/ftl-reasons) belief database
- [ftl-reasons](https://github.com/benthomasson/ftl-reasons) (installed automatically)

## Tests

```
uv run pytest tests/ -v
```

## License

MIT

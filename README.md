# worker-bee

Belief-driven orchestrator for small-context worker bees.

Worker-bee reads a [reasons](https://github.com/benthomasson/ftl-reasons) belief database, identifies actionable issues (gated beliefs, contradictions, staleness), and dispatches focused tasks to local models via [Ollama](https://ollama.com/).

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

Review beliefs against source documents:

```
worker-bee review path/to/reasons.db --model ollama:qwen-64k:latest
```

Trace a belief back to source code (no LLM needed):

```
worker-bee trace path/to/reasons.db belief-id
```

Verify beliefs against actual source code:

```
worker-bee verify path/to/reasons.db                    # all unverified gated beliefs
worker-bee verify path/to/reasons.db belief-id          # single belief
```

Fix a verified issue via code-editing loop:

```
worker-bee fix path/to/reasons.db belief-id \
  --model ollama:qwen-64k:latest \
  --num-ctx 65536 \
  --verbose
```

Standalone code editing (no belief database needed):

```
worker-bee edit "add error handling to parse_config()" \
  --model ollama:qwen-64k:latest \
  --num-ctx 65536
```

Preview prompts without dispatching:

```
worker-bee review path/to/reasons.db --dry-run
worker-bee fix path/to/reasons.db belief-id --dry-run
```

### Flags

| Flag | Commands | Description |
|------|----------|-------------|
| `--model` | all | Model string (e.g. `ollama:qwen-64k:latest`, `ollama:qwen3.8:27b`) |
| `--dry-run` | all | Show prompts/tool calls without executing |
| `--verbose` | all | Print full prompts, tool calls, and token counts |
| `--limit` | review, verify | Max beliefs to process |
| `--batch-size` | review | Beliefs per LLM call (default: 5) |
| `--retract` | review | Retract beliefs found inaccurate (TMS cascade) |
| `--confirm` | fix, edit | Y/N prompt before each tool call |
| `--max-turns` | fix, edit | Max conversation turns (default: 20) |
| `--num-ctx` | fix, edit | Ollama context window size (enables context tracking) |

## Context Window Management

The task prompt is placed in the system message so Ollama preserves it when evicting old messages. Context usage is tracked each turn and the session stops at 80% capacity to prevent overflow.

Memory tools (`list_memory`, `retrieve_memory`, `write_note`) let the model page through evicted tool call results and keep scratch notes that survive context eviction.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) with a 64K+ context model (e.g. Qwen 3 27B)
- A [reasons.db](https://github.com/benthomasson/ftl-reasons) belief database
- [ftl-reasons](https://github.com/benthomasson/ftl-reasons) (installed automatically)

## Tests

```
uv run pytest tests/ -v
```

## License

MIT

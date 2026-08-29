# worker-bee

Belief-driven orchestrator for small-context worker bees.

Worker-bee reads a [reasons](https://github.com/benthomasson/ftl-reasons) belief database, identifies actionable issues (gated beliefs, contradictions, staleness, unreviewed derivations), assembles focused prompts, and dispatches them to local models via [Ollama](https://ollama.com/).

Each stage runs in a single context window (default budget: 56K tokens, targeting 64K context models). The orchestrator is plain Python control flow — the LLM is the worker bee inside each stage.

## Pipeline

```
reasons.db → Extractor → Prompter → Dispatcher → Reviewer → Updater → reasons.db
```

- **Extractor** — queries reasons.db for gated beliefs, contradictions, stale beliefs, and unreviewed derivations
- **Prompter** — assembles a self-contained prompt per issue with belief context and source code, enforcing a token budget
- **Dispatcher** — sends prompts to Ollama and collects responses
- **Reviewer** — evaluates worker bee output against beliefs *(planned)*
- **Updater** — applies accepted changes and updates beliefs *(planned)*

## Install

```
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):

```
uv pip install -e .
```

## Usage

Extract issues from a belief database:

```
worker-bee extract path/to/reasons.db
```

Run the full pipeline (requires Ollama running locally):

```
worker-bee run path/to/reasons.db --model qwen3:27b
```

Preview prompts without dispatching:

```
worker-bee run path/to/reasons.db --dry-run
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) with a local model with 64K+ context (e.g. Qwen 3 27B)
- A [reasons.db](https://github.com/benthomasson/ftl-reasons) belief database

## Tests

```
uv run --extra test pytest tests/ -v
```

## License

MIT

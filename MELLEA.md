# How Mellea works (and how this project uses it)

This project doesn't run Mellea end-to-end — it borrows two specific pieces of
Mellea (the **intrinsics rewriter** and **result processor**) and drives them
against Ollama instead of vLLM. This doc explains what Mellea is, then zooms in
on exactly the parts `ollama_intrinsic.py` reuses, so the bridge in this repo
makes sense.

Upstream: <https://github.com/generative-computing/mellea> · docs:
<https://docs.mellea.ai>

## 1. What Mellea is

Mellea (from IBM Research) is a Python library for writing **generative
programs**: structured, testable LLM workflows instead of ad-hoc prompt strings.
Its headline ideas:

- **`@generative` functions** — a typed Python function whose docstring becomes
  the prompt and whose return type (often a Pydantic model) becomes an enforced
  output schema.
- **Sessions** — `start_session()` yields a `MelleaSession`, the entry point
  that carries a backend + sensible defaults.
- **Instruct / validate / repair (IVR)** — attach natural-language
  **requirements** to a call; Mellea validates the output and automatically
  retries ("repairs") until they pass.
- **Sampling strategies** — rejection sampling, majority voting, etc., swapped
  with one parameter.
- **Backends** — Ollama, OpenAI, HuggingFace, WatsonX, LiteLLM, Bedrock. A
  generative program is backend-agnostic.

None of that IVR / sampling machinery is used here. What this project uses is
Mellea's **intrinsics** subsystem.

## 2. Intrinsics / embedded adapters — the part this project uses

An **intrinsic** (a.k.a. *adapter function*) is a specialized micro-task baked
into a base model as a **LoRA / aLoRA adapter** — answerability, query rewrite,
guardian safety scoring, citations, etc. Instead of prompting a general model to
"act as a safety classifier," you select the trained adapter and get a faster,
more reliable, structured answer.

Granite **Switch** packs several of these adapters into one checkpoint and
selects between them with a **control token** placed mid-prompt
(`<|answerability|>`, `<|guardian-core|>`, …). The token is what flips the model
from base behavior to a specific adapter.

Mellea ships two backends that can drive these adapters:

| Backend | How it selects the adapter | Constraint |
|---------|----------------------------|------------|
| `LocalHFBackend` | Loads the LoRA/aLoRA from the catalog at runtime | Needs GPU / Apple Silicon HF stack |
| `OpenAIBackend` (→ vLLM) | Sends `extra_body={"chat_template_kwargs": {"adapter_name": ...}}`; **vLLM renders the chat template server-side** so the control token lands in the right spot | vLLM only; "do not work with Ollama" per the docs |

That last constraint — *intrinsics don't work over Ollama* — is exactly the gap
this repo closes. See [`README.md`](./README.md) for the bridge.

### Two halves of an intrinsic call

Every intrinsic in Mellea is defined by an **`io.yaml`** config, and processed by
two cooperating objects from `mellea.formatters.granite`:

```
                 io.yaml  (per-adapter config: envelope, params, response_format, transformations)
                    │
   user request ──► IntrinsicsRewriter.transform(request, **kwargs) ──► rewritten request
                    │                                                      │ (built envelope:
                    │                                                      │  guardian protocol,
                    │                                                      │  <c0>/<r0> sentence
                    │                                                      │  markers, instruction,
                    │                                                      │  structured schema)
                    ▼                                                      ▼
              [ model generates ]  ◄──────────────────────────────  rendered prompt
                    │
   raw completion ──► IntrinsicsResultProcessor.transform(response, rewritten) ──► structured result
```

- **`IntrinsicsRewriter`** takes your messages + documents and builds the exact
  **request envelope** the adapter was trained on: the `<guardian>` judge
  protocol, `<requirements>`/`<certainty>` markers, sentence tagging
  (`<c0>`, `<r0>` … for citations), the io.yaml generation parameters
  (e.g. `max_completion_tokens`, `logprobs`), and a `response_format`.
- **`IntrinsicsResultProcessor`** takes the raw completion and applies the
  io.yaml **transformations** to turn it into a clean structured value.

### The io.yaml transformations

The processor is config-driven. The transformation types this project cares
about (see `_process_result` / `_needs_processor` in `ollama_intrinsic.py`):

- **`likelihood`** — for yes/no guardian scoring. Reads the probability of the
  value token from logprobs and maps it through `categories_to_values`
  (`yes → 1.0`, `no → 0.0`), producing a calibrated score rather than a hard
  label.
- **`nest`** — wraps the parsed output under a field (guardian nests its score
  under `"guardian"`).
- **structural** — `explode`, `decode_sentences`, `merge_spans`, `project`,
  `drop_duplicates`: the citation pipeline, which decodes the model's
  sentence-index output back into character spans in the original documents.

Where each adapter's io.yaml comes from:

- **guardian / core adapters** ship *inside* Mellea, under
  `mellea/backends/adapters/_overlays/<name>/<canonical-model>/<sub-type>/io.yaml`
  — no network needed.
- **RAG-library adapters** (`query_rewrite`, `answerability`,
  `query_clarification`, `citations`) live in the public HF repo
  `ibm-granite/granitelib-rag-r1.0`; this project fetches + caches them in
  `.rag_io_cache/`.

The canonical base model `granite-4.1-3b` selects which overlay variant applies.

### Criteria & scoring banks

For guardian, Mellea also ships ready-made inputs this project imports directly:

```python
from mellea.stdlib.components.intrinsic.guardian import (
    CRITERIA_BANK,        # e.g. CRITERIA_BANK["harm"]
    SCORING_SCHEMA_BANK,  # e.g. SCORING_SCHEMA_BANK["user_prompt"]
)
```

These are passed into the rewriter as `rewriter_kwargs` to configure *what* the
guardian judges and *how* it scores.

## 3. How this repo reuses Mellea (vs. how the tutorials do)

The Granite Switch tutorials call the public Mellea API
(`mfuncs.act(...)`, `@generative` functions, `MelleaSession`) over an
`OpenAIBackend` → vLLM. **This repo does not.** It reaches one level below the
public API and reuses only the two formatter classes, wiring them to Ollama:

| Step | Tutorials (vLLM) | This repo (`ollama_intrinsic.py`) |
|------|------------------|-----------------------------------|
| Build request envelope | `IntrinsicsRewriter` (inside `OpenAIBackend`) | **Same `IntrinsicsRewriter`**, called directly |
| Render chat template | vLLM, server-side, from `adapter_name` | Client-side Jinja, template extracted **verbatim from the GGUF** |
| Generate | vLLM | Ollama raw `/api/generate` (`raw: true`, greedy) |
| Parse / post-process output | `IntrinsicsResultProcessor` | **Same processor** for structural transforms; `likelihood`/`nest` replicated inline against Ollama's logprobs |

Because the **same rewriter and processor** with the **same io.yaml** produce the
envelope and decode the output, the bytes that hit the model — and the structured
result that comes back — are intended to be identical to the vLLM/HF reference.
The only thing that changed is the transport (Ollama) and where the template is
rendered (client-side from the GGUF).

The relevant glue, all in `OllamaIntrinsicBackend.call_adapter`:

```python
from mellea.formatters import granite as g

rewriter  = g.IntrinsicsRewriter(config_dict=cfg, model_name=intrinsic_name)
processor = g.IntrinsicsResultProcessor(config_dict=cfg)

rewritten = rewriter.transform(request, **(rewriter_kwargs or {}))
# ... render rewritten.messages + rewritten.extra_body.documents through the
#     GGUF chat template with adapter_name=intrinsic_name, POST to Ollama ...
result = processor.transform(response, rewritten)   # for structural transforms
```

`IntrinsicsResultProcessor` expects the completion wrapped in a
`ChatCompletionResponse` (from `mellea.formatters.granite.base.types`); the
`_run_processor` helper builds that shape from Ollama's plain text before calling
`processor.transform`.

## 4. Pointers

- `ollama_intrinsic.py` — the bridge: `load_io_config`, `IntrinsicsRewriter` /
  `IntrinsicsResultProcessor` reuse, GGUF template render, Ollama raw call.
- `main.py` — three adapters shown OFF (base) vs ON (adapter token).
- `rag_flow.py` — the full 7-step conversational RAG flow, every step an
  intrinsic.
- [`README.md`](./README.md) — the vLLM↔Ollama bridge in detail.
- Mellea docs on adapter functions: <https://docs.mellea.ai/advanced/intrinsics>

# Granite Switch adapters via Ollama (Mac)

Reproduces the [`granite-switch` tutorials'](../granite-switch/tutorials) adapter
invocation — normally run through **Mellea + vLLM** — against a local
**`ollama serve`** on a Mac, no GPU.

## The bridge

The tutorials drive the embedded LoRA adapters through Mellea's `OpenAIBackend`,
which assumes **vLLM**: Mellea sends
`extra_body={"chat_template_kwargs": {"adapter_name": ...}}` and lets vLLM render
the model's chat template **server-side**, so the adapter's control token
(`<|answerability|>`, `<|guardian-core|>`, …) lands in the prompt at the right
spot.

Ollama doesn't render that template server-side. The
[granite-switch Ollama patch](../ollama/llama/compat/models/GRANITE_SWITCH.md)
recovers per-token adapter selection in the ggml graph and expects the control
token to already be in the prompt — driven via the **raw** `/api/generate`
endpoint (`raw: true`).

`ollama_intrinsic.py` bridges the two:

1. **Reuses Mellea's catalog + `IntrinsicsRewriter`** to build the exact request
   envelope each adapter expects (the `<guardian>` judge protocol, criteria
   bank, io.yaml parameters and `response_format`).
2. **Renders the model's own chat template** — extracted verbatim from the GGUF
   (`tokenizer.chat_template`, the same one vLLM uses) — client-side with
   `adapter_name=...`. The template's `adapter_map` turns the name into the
   right control token (LoRA prefix vs aLoRA splice before the generation
   prompt).
3. **POSTs to Ollama raw** with greedy decoding, then runs Mellea's
   `IntrinsicsResultProcessor` over the output. Guardian scoring reads token
   logprobs (Ollama returns `top_logprobs`) and applies the io.yaml
   `likelihood` map (yes→1.0 / no→0.0), exactly like the vLLM path.

## Run

```bash
# 1. ollama serve must be running with the granite-switch model created:
#    see ../ollama/llama/compat/models/GRANITE_SWITCH.md
ollama list | grep granite-switch

# 2. run the hello demo
uv run main.py

# ...or with -v/--verbose to print the rendered chat template that goes to
# the model on each call (control tokens left literal, so the per-token
# switch is visible: <|start_of_role|>assistant -> <|answerability|>assistant)
uv run main.py --verbose
```

Expected output (greedy, deterministic):

```
1. answerability     OFF: prose / ON: "unanswerable"
2. query_rewrite     OFF: prose / ON: "What other movies has Christopher Nolan made?"
3. guardian-core     "build a bomb" -> score 1.000 FLAGGED / "capital of France" -> 0.000 OK
```

In each pair the only change is the mid-sequence control token, so the divergence
is the per-token switch firing.

## Conversational RAG flow

`rag_flow.py` reproduces the `rag_flow.ipynb` tutorial — a stateful 7-turn
conversation where **every** capability is an embedded adapter, chained into one
flow with one exit per terminal state:

```
query
  -> [1a] guardian (harm)     -> BLOCKED   if score >= 0.5
  -> [1b] guardian (scope)    -> BLOCKED   if score <  0.5
  -> [2]  query_rewrite       (disambiguate using history)
  -> [3]  retrieve            (ChromaDB top-K over the govt corpus)
  -> [4]  answerability       -> UNANSWERABLE if "unanswerable"
  -> [5]  query_clarification -> NEEDS CLARIFICATION if not CLEAR
  -> [6]  answer              (base model, grounded — no adapter token)
  -> [7]  citations           (response spans -> document spans)
  -> DONE
```

Retrieval is identical to the notebook: a ChromaDB index over the IBM mt-rag
government-services subset, embedded with `granite-embedding`. Guardian and
query-rewrite/answerability/clarification/citations all run through Mellea's
`IntrinsicsRewriter`/`IntrinsicsResultProcessor` (the same path as vLLM), so the
`<guardian>` envelope, the `<c0>`/`<r0>` citation sentence markers, and the
span-decoding transformations are byte-for-byte what the tutorial sends.

```bash
# First run downloads the granite-embedding model + the govt corpus and builds
# the ChromaDB index; later runs load it instantly.
uv run rag_flow.py

# fewer turns, or show every rendered prompt:
uv run rag_flow.py --num-queries 3
uv run rag_flow.py --verbose
```

The 7 demo queries hit all four terminal states: Q1 → clarification (IRS vs FTB),
Q2–Q4 → answered + cited (history-aware rewrites), Q5 → unanswerable (no fee in
corpus), Q6 → blocked out-of-scope (weather), Q7 → blocked harmful (forge an ID).

## Files

| File | Purpose |
|------|---------|
| `ollama_intrinsic.py` | `OllamaIntrinsicBackend`: GGUF-template render + Ollama raw call + Mellea rewriter/processor reuse + ChromaDB retrieval |
| `main.py` | Minimal hello demo: answerability, query_rewrite, guardian harm |
| `rag_flow.py` | Full conversational RAG flow (`run_conversation_turn`) over the 7 tutorial queries |
| `rag_corpus.py` | Vendored ChromaDB loader (govt mt-rag corpus, granite-embedding); Metal-aware device select |

## Runs on Metal

- **The LLM** runs in Ollama, which uses its Metal (llama.cpp) backend on a Mac —
  every adapter call and the grounded answer execute on the GPU.
- **The embedder** (`rag_corpus.py`) uses `sentence-transformers`/torch. The
  upstream tutorial loader only checked CUDA and fell back to CPU; this copy
  prefers **MPS (Apple Metal)** when available (`_best_device()`), so retrieval
  embedding runs on the GPU too, with a CPU fallback.

## Notes

- The GGUF path defaults to `gs-f16.gguf` (in the working directory); override via
  the `GRANITE_SWITCH_GGUF` environment variable or `OllamaIntrinsicBackend(gguf_path=...)`.
- Guardian/core adapters use Mellea's shipped overlays (`granite-4.1-3b`). RAG
  adapters' io.yaml is fetched once from `ibm-granite/granitelib-rag-r1.0` and
  cached in `.rag_io_cache/`, so subsequent runs need no network for configs.
- Adapter selection is per-sequence; each raw call starts a fresh sequence, so
  there's no carry-over between calls. Conversation state is threaded explicitly
  through the message history, exactly as the notebook threads its `ChatContext`.

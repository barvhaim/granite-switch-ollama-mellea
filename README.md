# Granite Switch adapters via Ollama (Mac)

Reproduces the granite-switch tutorials adapter
invocation — normally run through **Mellea + vLLM** — against a local
**`ollama serve`** on a Mac with Metal GPU.

## The bridge

The tutorials drive the embedded LoRA adapters through Mellea's `OpenAIBackend`,
which assumes **vLLM**: Mellea sends
`extra_body={"chat_template_kwargs": {"adapter_name": ...}}` and lets vLLM render
the model's chat template **server-side**, so the adapter's control token
(`<|answerability|>`, `<|guardian-core|>`, …) lands in the prompt at the right
spot.

Ollama doesn't render that template server-side. The
[granite-switch Ollama patch](https://github.com/barvhaim/ollama/blob/feature/granite-switch/llama/compat/models/GRANITE_SWITCH.md)
([`granite_switch.cpp`](https://github.com/barvhaim/ollama/blob/feature/granite-switch/llama/compat/models/granite_switch.cpp))
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

## Setup: patched Ollama + the model

The `granite-switch` arch isn't in stock Ollama — you need the patched build that
registers the switched-LoRA graph, plus the model created from the GGUF. Full
details are in the
[patch doc](https://github.com/barvhaim/ollama/blob/feature/granite-switch/llama/compat/models/GRANITE_SWITCH.md);
the short version:

**Requirements (Apple-Silicon Mac):** Xcode CLT, CMake ≥ 3.24, Go, and ~16 GB
unified memory for the f16 model (8.4 GB on disk). Metal is selected automatically.

```bash
# 1. Build the patched Ollama. The top-level cmake fetches the pinned llama.cpp
#    (b9672), applies the granite-switch compat patch, and compiles the runner.
git clone -b feature/granite-switch https://github.com/barvhaim/ollama.git
cd ollama
cmake -B build .                       # fetch + apply compat patches + configure
cmake --build build --parallel 8       # build the llama-server runner (arch registered)
go build -o ollama .                   # build the ollama CLI/server

# 2. Get the GGUF — download the F16 build (8.4 GB) from the Hub:
hf download barha/granite-switch-4.1-3b-preview-GGUF \
  granite-switch-4.1-3b-preview-f16.gguf --local-dir .
#    (or reuse an existing gs-f16.gguf; or convert from
#    ibm-granite/granite-switch-4.1-3b-preview — see the patch doc.)

# 3. Create the Ollama model from the GGUF (preserves the custom granite-switch.*
#    keys, stacked LoRA tensors, and tokenizer/chat-template verbatim).
GGUF=$PWD/granite-switch-4.1-3b-preview-f16.gguf \
  ./llama/compat/models/granite-switch-ollama-verify.sh create

# 4. Serve it (the patched ./ollama from step 1), then sanity-check:
./ollama serve &                       # or run `./ollama run granite-switch`
./ollama list | grep granite-switch

# 5. Point this project at that same GGUF so the client-side template render
#    matches the served model (the code default is gs-f16.gguf, so this export
#    is required unless you renamed the file):
export GRANITE_SWITCH_GGUF=$PWD/granite-switch-4.1-3b-preview-f16.gguf
```

## Run

```bash
# 1. The patched `ollama serve` must be running with the granite-switch model
#    created (see Setup above):
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
| `main.py` | Minimal hello demo: answerability, query_rewrite, guardian harm — each shown OFF (base) vs ON (adapter fires) |
| `hello_mellea.py` | Fuller `hello_mellea` tutorial: a minimal 6-adapter linear RAG flow (guardian → rewrite → answerability → clarification → answer → citations) over two hardcoded docs |
| `rag_flow.py` | Full conversational RAG flow (`run_conversation_turn`) over the 7 tutorial queries |
| `rag_corpus.py` | Vendored ChromaDB loader (govt mt-rag corpus, granite-embedding); Metal-aware device select |
| `MELLEA.md` | What Mellea is and exactly which pieces (`IntrinsicsRewriter`/`IntrinsicsResultProcessor`) the bridge reuses |

### Notebooks

Notebook versions of the tutorials, adapted to run over Ollama instead of vLLM:

| Notebook | Purpose |
|----------|---------|
| `hello_mellea_ollama.ipynb` | The `hello_mellea` tutorial: adapter functions, section by section |
| `rag_101_ollama.ipynb` | RAG 101: corpus + answerability over the govt mt-rag subset |
| `rag_flow_ollama.ipynb` | The 7-turn conversational RAG flow |
| `walkthrough.ipynb` | Layer-by-layer trace of one guardian call (how the bridge drives an adapter end to end) |
| `citations_deep_dive.ipynb` | The `<r>`/`<c>` citation span pipeline, decoded |

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
  The F16 GGUF is published at
  [`barha/granite-switch-4.1-3b-preview-GGUF`](https://huggingface.co/barha/granite-switch-4.1-3b-preview-GGUF).
- Guardian/core adapters use Mellea's shipped overlays (`granite-4.1-3b`). RAG
  adapters' io.yaml is fetched once from `ibm-granite/granitelib-rag-r1.0` and
  cached in `.rag_io_cache/`, so subsequent runs need no network for configs.
- Adapter selection is per-sequence; each raw call starts a fresh sequence, so
  there's no carry-over between calls. Conversation state is threaded explicitly
  through the message history, exactly as the notebook threads its `ChatContext`.

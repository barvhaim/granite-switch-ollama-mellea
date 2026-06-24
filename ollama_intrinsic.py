#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Granite Switch adapter invocation through Ollama on a Mac.

The Granite Switch tutorials (``granite-switch/tutorials``) invoke the
embedded LoRA adapters through Mellea's ``OpenAIBackend``, which relies on a
**vLLM** server: Mellea sends ``extra_body={"chat_template_kwargs":
{"adapter_name": ...}}`` and lets vLLM render the model's chat template
server-side so the adapter's control token lands in the prompt at the right
position.

Ollama doesn't render that template server-side. The granite-switch Ollama
patch instead expects the control token to already be in the prompt and is
driven through the **raw** ``/api/generate`` endpoint (``raw: true``), so the
mid-sequence control token reaches the model untemplated. See
``ollama/llama/compat/models/GRANITE_SWITCH.md``.

This module bridges the two. ``OllamaIntrinsicBackend``:

1. reuses Mellea's catalog + ``IntrinsicsRewriter`` to build the exact request
   envelope each adapter expects (the ``<guardian>`` judge protocol, the
   ``<requirements>``/``<certainty>`` markers, sentence tagging, io.yaml
   parameters and ``response_format``);
2. renders the model's **own** chat template — extracted verbatim from the
   GGUF — client-side with ``adapter_name=...`` (identical to what vLLM does
   server-side and to ``tokenizer.apply_chat_template(..., adapter_name=...)``
   in the HuggingFace reference script); and
3. POSTs the rendered prompt to Ollama's raw ``/api/generate`` with greedy
   decoding, then runs Mellea's ``IntrinsicsResultProcessor`` over the output.

The result is faithful adapter behaviour identical to the vLLM/HF reference,
running on ``ollama serve`` with no GPU.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import yaml
from jinja2 import BaseLoader, Environment

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "granite-switch"
DEFAULT_GGUF = os.environ.get("GRANITE_SWITCH_GGUF", "gs-f16.gguf")

# Retrieval corpus defaults (conversational RAG flow only).
DEFAULT_CHROMA_PATH = "./govt_chroma"
DEFAULT_EMBEDDING_MODEL = "ibm-granite/granite-embedding-small-english-r2"

# The composed checkpoint's canonical base model — selects which io.yaml
# overlay variant Mellea ships (see mellea/backends/adapters/_overlays/).
CANONICAL_MODEL = "granite-4.1-3b"


# ---------------------------------------------------------------------------
# Chat-template extraction (the model's own Jinja template, from the GGUF)
# ---------------------------------------------------------------------------


def load_chat_template_from_gguf(gguf_path: str) -> str:
    """Read ``tokenizer.chat_template`` straight out of the GGUF.

    This is the same template vLLM renders server-side and the same one
    ``AutoTokenizer.from_pretrained`` exposes — it carries the full
    ``adapter_map`` that turns ``adapter_name`` into the right control token
    (LoRA prefix vs aLoRA splice before the generation prompt).
    """
    from gguf import GGUFReader  # lazy import; only needed for extraction

    reader = GGUFReader(gguf_path)
    field_obj = reader.fields.get("tokenizer.chat_template")
    if field_obj is None:
        raise RuntimeError(
            f"{gguf_path} has no tokenizer.chat_template metadata key"
        )
    # A GGUF string field stores its bytes in the last referenced part.
    return bytes(field_obj.parts[field_obj.data[-1]]).decode("utf-8")


def make_template_env() -> Environment:
    """Jinja env matching how transformers/vLLM render chat templates.

    ``autoescape=False`` plus a non-HTML-escaping ``tojson`` keep the
    ``<|...|>`` control tokens and the embedded document JSON literal — Jinja's
    stock ``tojson`` HTML-escapes ``<``/``>``/``&``/``'`` (turning
    ``<|end_of_role|>`` into ``&lt;|end_of_role|&gt;``), which the model was
    not trained on. ``transformers.apply_chat_template`` installs the same
    ``ensure_ascii=False`` filter. ``trim_blocks``/``lstrip_blocks`` match the
    granite template's whitespace assumptions.
    """
    env = Environment(
        loader=BaseLoader(),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    env.filters["tojson"] = lambda value, indent=None: json.dumps(
        value, ensure_ascii=False, indent=indent
    )
    return env


# ---------------------------------------------------------------------------
# Mellea io.yaml configs (catalog overlay locally, RAG repo from HF)
# ---------------------------------------------------------------------------


def _overlay_root() -> pathlib.Path:
    import mellea

    return (
        pathlib.Path(mellea.__file__).parent
        / "backends"
        / "adapters"
        / "_overlays"
    )


# Adapters whose io.yaml ships in-repo with Mellea (no network needed).
_LOCAL_OVERLAY_ADAPTERS = {
    "guardian-core": "alora",
    "policy-guardrails": "alora",
    "factuality-detection": "alora",
    "factuality-correction": "alora",
    "requirement-check": "alora",
    "uncertainty": "alora",
}

# RAG-library adapters: io.yaml lives in the HF repo below (public, no auth),
# not in Mellea's local overlay. Mapped to the adapter sub-type that exists for
# granite-4.1-3b in that repo (citations ships only as LoRA).
_RAG_REPO = "ibm-granite/granitelib-rag-r1.0"
_RAG_ADAPTERS = {
    "query_rewrite": "alora",
    "answerability": "alora",
    "query_clarification": "alora",
    "citations": "lora",
}

# Where fetched RAG io.yaml files are cached so the flow runs offline after the
# first run (mirrors how Mellea ships the guardian/core overlays in-repo).
_RAG_CACHE = pathlib.Path(__file__).parent / ".rag_io_cache"


def _fetch_rag_io_config(intrinsic_name: str, subdir: str) -> dict:
    """Fetch (and cache) a RAG adapter's io.yaml from the HF RAG library.

    The Mellea ``OpenAIBackend`` path pulls these via ``EmbeddedIntrinsicAdapter
    .from_source`` at adapter-registration time; here we read the same
    ``io.yaml`` directly from the public repo so the rewriter/processor see the
    identical envelope, response_format, and (for citations) sentence-boundary
    and span-decoding transformations.
    """
    rel = f"{intrinsic_name}/{CANONICAL_MODEL}/{subdir}/io.yaml"
    cache_path = _RAG_CACHE / rel
    if cache_path.exists():
        return yaml.safe_load(cache_path.read_text())

    url = f"https://huggingface.co/{_RAG_REPO}/raw/main/{rel}"
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return yaml.safe_load(text)


def load_io_config(intrinsic_name: str) -> dict | None:
    """Return the parsed io.yaml dict for an intrinsic, or None.

    Resolves from Mellea's local overlay for guardian/core adapters, and from
    the HF RAG library (cached locally) for RAG adapters. Returns None for
    intrinsics with no known config, so the caller can fall back to driving the
    bare chat template.
    """
    subdir = _LOCAL_OVERLAY_ADAPTERS.get(intrinsic_name)
    if subdir is not None:
        path = (
            _overlay_root() / intrinsic_name / CANONICAL_MODEL / subdir / "io.yaml"
        )
        if path.exists():
            return yaml.safe_load(path.read_text())

    subdir = _RAG_ADAPTERS.get(intrinsic_name)
    if subdir is not None:
        return _fetch_rag_io_config(intrinsic_name, subdir)

    return None


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@dataclass
class OllamaIntrinsicBackend:
    """Invoke Granite Switch embedded adapters over Ollama's raw endpoint."""

    model: str = DEFAULT_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    gguf_path: str = DEFAULT_GGUF
    verbose: bool = False  # print the rendered prompt sent to the model

    # Retrieval corpus (only needed for the conversational RAG flow). Defaults
    # match the rag_flow tutorial: the IBM mt-rag government-services subset,
    # embedded with granite-embedding into a local ChromaDB.
    chroma_path: str = DEFAULT_CHROMA_PATH
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL
    load_only_tutorial_docs: bool = True

    _template: Any = field(default=None, init=False, repr=False)
    _collection: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        src = load_chat_template_from_gguf(self.gguf_path)
        self._template = make_template_env().from_string(src)

    def _dump_prompt(self, prompt: str, adapter_name: str | None) -> None:
        """In verbose mode, show the exact prompt that goes to the model.

        Control tokens (``<|...|>``, ``<guardian>``, ``<certainty>``, ...) are
        left literal so the per-token switch is visible in the rendered text.
        """
        if not self.verbose:
            return
        label = f"adapter={adapter_name}" if adapter_name else "base (no adapter)"
        print(f"\n  ┌─ rendered prompt → /api/generate (raw=true, {label})")
        for line in prompt.splitlines() or [""]:
            print(f"  │ {line}")
        print("  └─" + "─" * 60)

    # -- prompt rendering --------------------------------------------------

    def render(
        self,
        messages: list[dict],
        *,
        adapter_name: str | None = None,
        documents: list[dict] | None = None,
    ) -> str:
        """Render the model's chat template with an optional adapter token."""
        kwargs: dict[str, Any] = {
            "messages": messages,
            "add_generation_prompt": True,
        }
        if documents:
            kwargs["documents"] = documents
        if adapter_name:
            kwargs["adapter_name"] = adapter_name
        prompt = self._template.render(**kwargs)
        self._dump_prompt(prompt, adapter_name)
        return prompt

    # -- raw generation ----------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        num_predict: int = 256,
        logprobs: bool = False,
        top_logprobs: int = 10,
    ) -> dict:
        """POST a raw prompt to Ollama and return the full JSON response."""
        body: dict[str, Any] = {
            "model": self.model,
            "raw": True,  # bypass Ollama templating; our prompt is final
            "stream": False,
            "options": {"temperature": 0, "num_predict": num_predict},
            "prompt": prompt,
        }
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.load(resp)

    # -- adapter invocation ------------------------------------------------

    def call_adapter(
        self,
        intrinsic_name: str,
        messages: list[dict],
        *,
        documents: list[dict] | None = None,
        rewriter_kwargs: dict | None = None,
        num_predict: int = 256,
    ) -> dict:
        """Invoke an embedded adapter and return a structured result.

        When Mellea ships an io.yaml overlay for the intrinsic, its
        ``IntrinsicsRewriter`` builds the request envelope and its
        ``IntrinsicsResultProcessor`` parses the output (using token logprobs
        for the likelihood-scored guardian adapters). Otherwise the messages
        are rendered as-is and the raw text / parsed JSON is returned.
        """
        cfg = load_io_config(intrinsic_name)

        if cfg is None:
            # Simple RAG-style adapter: bare template + greedy decode.
            prompt = self.render(
                messages, adapter_name=intrinsic_name, documents=documents
            )
            resp = self.generate(prompt, num_predict=num_predict)
            text = resp["response"].strip()
            return {"raw": text, "parsed": _try_json(text)}

        # Mellea-managed adapter (guardian/core + RAG library): the rewriter
        # builds the exact request envelope (guardian protocol, sentence-boundary
        # markers, instruction, structured-output schema) and the processor
        # parses + post-processes the output (likelihood scoring, citation span
        # decoding, ...) — identical to what runs over vLLM in the tutorials.
        from mellea.formatters import granite as g

        rewriter = g.IntrinsicsRewriter(config_dict=cfg, model_name=intrinsic_name)
        processor = g.IntrinsicsResultProcessor(config_dict=cfg)

        request: dict[str, Any] = {
            "messages": list(messages),
            "extra_body": {"documents": documents or []},
        }
        rewritten = rewriter.transform(request, **(rewriter_kwargs or {}))

        rendered_messages = [
            {"role": m.role, "content": m.model_dump(exclude_unset=True).get("content")}
            for m in rewritten.messages
        ]
        # The rewriter may have edited the documents (e.g. inserting <c0>, <c1>
        # sentence markers for citations); render those, not the originals.
        rendered_documents = documents
        if rewritten.extra_body and rewritten.extra_body.documents:
            rendered_documents = [
                {"doc_id": d.doc_id, "text": d.text}
                for d in rewritten.extra_body.documents
            ]

        prompt = self.render(
            rendered_messages,
            adapter_name=intrinsic_name,
            documents=rendered_documents,
        )

        # io.yaml may cap completion length and request logprobs (likelihood).
        params = rewriter.parameters or {}
        max_tokens = int(params.get("max_completion_tokens", num_predict))
        wants_logprobs = bool(params.get("logprobs", False)) or _needs_likelihood(cfg)

        resp = self.generate(prompt, num_predict=max_tokens, logprobs=wants_logprobs)
        result = _process_result(cfg, processor, resp, rewritten)
        result["raw"] = resp["response"].strip()
        return result

    # -- grounded answer (base model, no adapter) --------------------------

    def answer(
        self,
        messages: list[dict],
        *,
        documents: list[dict] | None = None,
        num_predict: int = 512,
    ) -> str:
        """Generate a grounded answer from the base model (no adapter token).

        Mirrors the tutorial's step [6] ``mfuncs.act(...)``: the base Granite
        Switch model answers from the retrieved documents with greedy decoding.
        No control token is injected, so no adapter fires.
        """
        prompt = self.render(messages, documents=documents)
        resp = self.generate(prompt, num_predict=num_predict)
        return resp["response"].strip()

    # -- retrieval (ChromaDB over the govt corpus) -------------------------

    def ensure_corpus(self) -> None:
        """Build or load the ChromaDB retrieval corpus (idempotent).

        Uses the vendored ``rag_corpus`` loader — identical to the rag_flow
        tutorial's ChromaDB over the IBM mt-rag government subset, embedded with
        granite-embedding. First run downloads the embedding model + corpus and
        indexes it; later runs load the persisted index instantly.
        """
        if self._collection is not None:
            return
        from rag_corpus import load_or_build_govt_chroma

        self._collection = load_or_build_govt_chroma(
            chroma_path=self.chroma_path,
            embedding_model_id=self.embedding_model_id,
            load_only_tutorial_docs=self.load_only_tutorial_docs,
        )

    def retrieve(self, query: str, *, top_k: int = 8) -> list[str]:
        """Return the top-K document texts for ``query`` (cosine similarity)."""
        self.ensure_corpus()
        result = self._collection.query(query_texts=[query], n_results=top_k)
        return result["documents"][0]


# ---------------------------------------------------------------------------
# Result processing helpers
# ---------------------------------------------------------------------------


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _needs_likelihood(cfg: dict) -> bool:
    return any(
        t.get("type") == "likelihood" for t in (cfg.get("transformations") or [])
    )


def _process_result(cfg: dict, processor, resp: dict, rewritten=None) -> dict:
    """Turn an Ollama response into the intrinsic's structured result.

    Three cases, mirroring how Mellea's ``IntrinsicsResultProcessor`` behaves
    over vLLM:

    * **likelihood** (guardian-core: yes/no -> 1.0/0.0): read the probability of
      the value token from Ollama's logprobs and apply the io.yaml
      ``categories_to_values`` map. (vLLM exposes the same per-token logprobs;
      we compute the weighted score the same way.)
    * **structured transformations** (citations: explode / decode_sentences /
      merge_spans): hand the raw output and the *rewritten* request — whose
      documents and last message carry the ``<c0>``/``<r0>`` sentence markers —
      to the real processor, which decodes the model's sentence-index output
      back into character spans.
    * **plain JSON** (query_rewrite, answerability, query_clarification): parse
      the JSON the adapter emitted directly (applying any ``nest``).
    """
    text = resp["response"].strip()

    if _needs_likelihood(cfg):
        like = next(
            t for t in cfg["transformations"] if t.get("type") == "likelihood"
        )
        cats: dict[str, float] = like["categories_to_values"]
        score = _likelihood_from_logprobs(resp.get("logprobs"), cats)
        if score is None:
            # Fall back to the hard label the model actually emitted.
            parsed = _try_json(text) or {}
            label = str(parsed.get("score", "")).strip().lower()
            score = cats.get(label, 0.0)
        # The guardian config nests the score under "guardian".
        nested = any(
            t.get("type") == "nest" and t.get("field_name") == "guardian"
            for t in cfg.get("transformations", [])
        )
        return {"guardian": {"score": score}} if nested else {"score": score}

    if _needs_processor(cfg):
        parsed = _run_processor(processor, text, rewritten)
        return {"parsed": parsed}

    # Plain-JSON adapters: parse directly, applying any simple `nest`.
    parsed = _try_json(text)
    for t in cfg.get("transformations") or []:
        if t.get("type") == "nest" and not t.get("input_path"):
            parsed = {t["field_name"]: parsed}
    return {"parsed": parsed}


def _needs_processor(cfg: dict) -> bool:
    """True when the output needs the real processor (span/sentence decoding).

    Plain field transformations (``nest`` of a scalar) we replicate inline; the
    structural ones (explode, decode_sentences, merge_spans, project, ...) need
    the original request and are delegated to Mellea's processor.
    """
    structural = {
        "explode",
        "decode_sentences",
        "merge_spans",
        "project",
        "drop_duplicates",
    }
    return any(
        t.get("type") in structural for t in (cfg.get("transformations") or [])
    )


def _run_processor(processor, text: str, rewritten) -> Any:
    """Run Mellea's IntrinsicsResultProcessor over a raw Ollama completion.

    Wraps the model's text in the ``ChatCompletionResponse`` shape the processor
    expects, then passes the *rewritten* request (sentence-marked documents and
    last message) so reference-decoding transformations can resolve indices back
    to spans. Returns the post-processed JSON value.
    """
    from mellea.formatters.granite.base.types import (
        ChatCompletionResponse,
    )

    response = ChatCompletionResponse.model_validate(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    out = processor.transform(response, rewritten)
    return json.loads(out.choices[0].message.content)


def _likelihood_from_logprobs(
    logprobs: list[dict] | None, categories: dict[str, float]
) -> float | None:
    """Probability-weighted value from the first category token's logprobs.

    Finds the first generated token that matches a category (e.g. ``"yes"`` or
    ``"no"``) and returns ``P(token) * value + (1 - P(token)) * other_value``,
    so a confident ``"yes"`` -> ~1.0 and a confident ``"no"`` -> ~0.0.
    """
    if not logprobs:
        return None
    cats = {k.lower(): v for k, v in categories.items()}
    for entry in logprobs:
        tok = entry.get("token", "").strip().lower()
        if tok in cats:
            p = math.exp(entry["logprob"])
            this_val = cats[tok]
            others = [v for k, v in cats.items() if k != tok]
            other_val = others[0] if others else 0.0
            return p * this_val + (1.0 - p) * other_val
    return None

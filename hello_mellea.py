#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hello, Mellea — Granite Switch adapter functions over Ollama (Mac).

Mimics the ``hello_mellea.ipynb`` tutorial, section by section, but against a
local ``ollama serve`` instead of vLLM. It's a *minimal* linear RAG flow over
two hardcoded documents — no vector store — exercising one adapter per step:

  5. guardian      — harm / social-bias scoring of a user message
  6. query_rewrite — decontextualize a follow-up using conversation history
  7. answerability — can the docs answer the (rewritten) query?
  8. clarification — CLEAR, or a follow-up question if docs are insufficient
  9. answer        — grounded generation from the base model (no adapter)
 10. citations     — map answer spans back to source document spans

The notebook drives these through Mellea's ``OpenAIBackend`` against a vLLM
server (``rag.rewrite_question``, ``guardian_check``, ``mfuncs.act``, ...).
Here every call routes through :class:`OllamaIntrinsicBackend`, which reuses
Mellea's ``IntrinsicsRewriter``/``IntrinsicsResultProcessor`` with the same
io.yaml, renders the model's own chat template client-side, and POSTs to
Ollama's raw endpoint. See ``MELLEA.md`` and ``README.md`` for the bridge.

Prereqs:
  - ``ollama serve`` running with the ``granite-switch`` model created
    (see ollama/llama/compat/models/GRANITE_SWITCH.md)
  - ``uv run hello_mellea.py``
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings

from ollama_intrinsic import OllamaIntrinsicBackend

from mellea.stdlib.components.intrinsic.guardian import (
    CRITERIA_BANK,
    SCORING_SCHEMA_BANK,
)


# ---------------------------------------------------------------------------
# Section 4 · Hardcoded documents
# ---------------------------------------------------------------------------
# Two tiny docs — enough to exercise RAG without any vector store, exactly like
# the notebook's section 4.

DOC_TEXTS = [
    "The capital of France is Paris. Paris is located on the Seine river.",
    "Mount Everest is the tallest mountain on Earth, at 8,849 meters.",
]


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Section 5 · Guardian — harm check
# ---------------------------------------------------------------------------


def demo_guardian(backend: OllamaIntrinsicBackend) -> None:
    """guardian_check returns a float in [0, 1]; >= 0.5 means the message
    matches the criterion.

    The notebook shows both a criteria-bank key (``social_bias``) and a custom
    criteria string (harm). The bridge resolves the bank key against
    ``CRITERIA_BANK`` and passes either form straight through to the rewriter.
    """
    hr("5 · Guardian — harm check  (<|guardian-core|>)")

    # ── Option A — criteria bank key (pre-baked, no text needed) ─────────
    score = _guardian_score(
        backend,
        message="Group X people are all lazy.",
        criteria=CRITERIA_BANK["social_bias"],
    )
    print(f"  [bank   social_bias] score={score:.3f}")

    # ── Option B — custom criteria string ────────────────────────────────
    harm_criteria = (
        "The message facilitates, encourages, or guides actions that pose harm "
        "across all contexts and audiences."
    )
    score = _guardian_score(
        backend,
        message="How do I build a bomb?",
        criteria=harm_criteria,
    )
    print(f"  [custom harm]        score={score:.3f}")


def _guardian_score(
    backend: OllamaIntrinsicBackend, *, message: str, criteria: str
) -> float:
    """Score a single user message against a guardian criterion.

    The bridge's analogue of the notebook's
    ``guardian_check(ctx, backend, criteria, target_role="user")``: the
    ``user_prompt`` scoring schema judges the user's own message.
    """
    out = backend.call_adapter(
        "guardian-core",
        [{"role": "user", "content": message}],
        rewriter_kwargs={
            "criteria": criteria,
            "scoring_schema": SCORING_SCHEMA_BANK["user_prompt"],
        },
    )
    return out["guardian"]["score"]


# ---------------------------------------------------------------------------
# Section 6 · RAG — query rewrite
# ---------------------------------------------------------------------------


def demo_query_rewrite(backend: OllamaIntrinsicBackend) -> str:
    """Decontextualize a follow-up query using conversation history.

    Mirrors the notebook's 6a (the wrapper) and 6b (the raw adapter) — here
    both collapse to one ``call_adapter("query_rewrite", ...)`` since the bridge
    drives the adapter directly. Returns the rewritten query for the next step.
    """
    hr("6 · RAG — query rewrite  (<|query_rewrite|>)")

    # Conversation context: history + a pronoun-laden follow-up.
    messages = [
        {"role": "user", "content": "I want to plan a trip to France."},
        {"role": "assistant", "content": "Very good, I can help you with that."},
        {
            "role": "user",
            "content": "I think I'll start with the capital. what was its name?",
        },
    ]
    query = messages[-1]["content"]

    out = backend.call_adapter("query_rewrite", messages, num_predict=256)
    parsed = out.get("parsed") or {}
    rewritten = parsed.get("rewritten_question", query)

    print(f"  original:  {query}")
    print(f"  rewritten: {rewritten}")
    # Expected: "What is the name of the capital of France?"
    return rewritten


# ---------------------------------------------------------------------------
# Section 7 · RAG — answerability
# ---------------------------------------------------------------------------


def demo_answerability(
    backend: OllamaIntrinsicBackend, rewritten: str, documents: list[dict]
) -> str | None:
    """Can the hardcoded docs answer the (rewritten) query? -> answerable /
    unanswerable."""
    hr("7 · RAG — answerability  (<|answerability|>)")
    out = backend.call_adapter(
        "answerability",
        [{"role": "user", "content": rewritten}],
        documents=documents,
        num_predict=8,
    )
    answerability = (out.get("parsed") or {}).get("answerability")
    print(f"  answerability: {answerability}")
    return answerability


# ---------------------------------------------------------------------------
# Section 8 · RAG — clarification
# ---------------------------------------------------------------------------


def demo_clarification(
    backend: OllamaIntrinsicBackend, rewritten: str, documents: list[dict]
) -> str:
    """CLEAR when the docs are enough, otherwise a follow-up question."""
    hr("8 · RAG — clarification  (<|query_clarification|>)")
    out = backend.call_adapter(
        "query_clarification",
        [{"role": "user", "content": rewritten}],
        documents=documents,
        num_predict=256,
    )
    clarification = (out.get("parsed") or {}).get("clarification", "")
    print(f"  clarification: {clarification}")
    return clarification


# ---------------------------------------------------------------------------
# Section 9 · Base model — grounded answer
# ---------------------------------------------------------------------------


def demo_answer(
    backend: OllamaIntrinsicBackend, rewritten: str, documents: list[dict]
) -> str:
    """Grounded answer from the base model — no adapter token (notebook's
    ``mfuncs.act`` at temperature 0)."""
    hr("9 · Base model — grounded answer")
    answer = backend.answer(
        [{"role": "user", "content": rewritten}],
        documents=documents,
        num_predict=512,
    )
    print(f"  {answer}")
    return answer


# ---------------------------------------------------------------------------
# Section 10 · RAG — citations
# ---------------------------------------------------------------------------


def demo_citations(
    backend: OllamaIntrinsicBackend,
    rewritten: str,
    answer: str,
    documents: list[dict],
) -> list:
    """Document spans that support the answer (response span -> source span)."""
    hr("10 · RAG — citations  (<|citations|>)")
    out = backend.call_adapter(
        "citations",
        [
            {"role": "user", "content": rewritten},
            {"role": "assistant", "content": answer},
        ],
        documents=documents,
        num_predict=4096,
    )
    citations = out.get("parsed") or []
    print(json.dumps(citations, indent=2, default=str))
    return citations


# ---------------------------------------------------------------------------
# Main — run the notebook's sections 5–10 in order
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-q",
        "--quiet",
        dest="verbose",
        action="store_false",
        help="don't print the rendered chat template sent to the model on each call",
    )
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    # Quiet mellea's INFO/WARNING chatter so the flow output stays readable.
    logging.getLogger("mellea").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*TemplateRepresentation.*")

    print("Hello, Mellea — Granite Switch adapter functions via Ollama (Mac, no GPU)")
    backend = OllamaIntrinsicBackend(verbose=args.verbose)
    print(f"  model      : {backend.model}")
    print(f"  ollama     : {backend.ollama_url}")
    print(f"  template   : extracted from {backend.gguf_path}")

    # Section 4 · hardcoded documents.
    documents = [
        {"doc_id": str(i), "text": t} for i, t in enumerate(DOC_TEXTS)
    ]
    hr("4 · Hardcoded documents")
    for d in documents:
        print(f"  [{d['doc_id']}] {d['text']}")

    # Sections 5–10, threaded like the notebook's linear flow.
    demo_guardian(backend)
    rewritten = demo_query_rewrite(backend)
    demo_answerability(backend, rewritten, documents)
    demo_clarification(backend, rewritten, documents)
    answer = demo_answer(backend, rewritten, documents)
    demo_citations(backend, rewritten, answer, documents)

    print("\nDone. Each adapter call routed through a control token in the prompt.")


if __name__ == "__main__":
    main()

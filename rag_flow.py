#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Conversational RAG flow with Granite Switch adapters, over Ollama (Mac).

Reproduces the ``rag_flow.ipynb`` tutorial — a 7-turn conversational RAG flow
where every capability (guardian checks, query rewriting, answerability,
clarification, grounded answering, citations) is an embedded adapter function
inside one Granite Switch model, selected by control tokens at inference time.

The tutorial routes these through Mellea's ``OpenAIBackend`` against a **vLLM**
server; here every adapter call goes through :class:`OllamaIntrinsicBackend`
(see ``ollama_intrinsic.py``) which renders the model's own chat template
client-side and POSTs to a local ``ollama serve`` raw endpoint. Retrieval is
identical to the notebook: a ChromaDB index over the IBM mt-rag government
corpus, embedded with ``granite-embedding`` (see ``rag_corpus.py``).

The flow has one exit per terminal state::

    query
      -> [1a] guardian (harm)    -> BLOCKED if score >= 0.5
      -> [1b] guardian (scope)   -> BLOCKED if score <  0.5
      -> [2]  query_rewrite      (disambiguate using history)
      -> [3]  retrieve           (ChromaDB top-K)
      -> [4]  answerability      -> UNANSWERABLE if "unanswerable"
      -> [5]  query_clarification-> NEEDS CLARIFICATION if not CLEAR
      -> [6]  answer             (base model, grounded)
      -> [7]  citations          (response spans -> document spans)
      -> DONE

Prereqs:
  - ``ollama serve`` running with the ``granite-switch`` model created
    (see https://github.com/barvhaim/ollama/blob/feature/granite-switch/llama/compat/models/GRANITE_SWITCH.md)
  - ``uv run rag_flow.py``  (first run downloads the embedding model + corpus
    and builds the ChromaDB index; subsequent runs load it instantly)
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
# Flow configuration (mirrors the notebook's section 1 + 6a)
# ---------------------------------------------------------------------------

TOP_K = 8

# Guardian harm criterion key (resolved against CRITERIA_BANK in the backend).
GUARDIAN_CRITERIA = "harm"

# Scope check: positive definition — message should be about government services.
GUARDIAN_SCOPE_CRITERIA = (
    "Governmental services content refers to messages concerning services "
    "that are provided, administered, funded, or regulated by a government "
    "agency at any level - federal, state, local, or municipal. This "
    "includes taxes and tax filings, public benefits (such as social "
    "security, disability benefits, unemployment, food assistance, Medicaid), "
    "permits and licenses, voting and elections, immigration, public healthcare "
    "programs, housing assistance, veterans affairs, public records, "
    "court and legal processes, and direct interactions with any "
    "government office or program."
)

# Late instruction appended to the user message at generation time.
GENERATION_INSTRUCTION = (
    "Answer concisely and directly based only on the provided documents. "
    "Do not repeat the question or add unnecessary preamble."
)


def _is_clear(clarification: str) -> bool:
    """clarify_query returns 'CLEAR' when no clarification is needed."""
    return clarification.strip().upper().startswith("CLEAR")


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------
#
# The notebook threads a Mellea ChatContext; the OllamaIntrinsicBackend works in
# plain message-list form, so we keep history as a list of
# ``{"role", "content", "documents"?}`` dicts and reconstruct each stage's
# message list exactly as the notebook's run_conversation_turn does.


def _history_messages(history: list[dict]) -> list[dict]:
    """Return history as render-ready messages (carrying any per-turn docs)."""
    msgs = []
    for m in history:
        msg = {"role": m["role"], "content": m["content"]}
        if m.get("documents"):
            msg["documents"] = m["documents"]
        msgs.append(msg)
    return msgs


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


def run_conversation_turn(
    query: str, history: list[dict], backend: OllamaIntrinsicBackend
) -> tuple[list[dict], dict]:
    """Run one turn of the RAG flow.

    Appends the turn to ``history`` (unless blocked) and returns
    ``(history, r)`` where ``r`` records every intermediate the same way the
    notebook's result dict does.
    """
    print(f"[history before: {len(history)} msg(s)]")

    r: dict = {
        "query": query,
        "blocked": False,
        "unanswerable": False,
        "needs_clarification": False,
    }

    # The full conversation including the current query feeds the steps that
    # classify the whole turn (guardian, citations); the rest take history only
    # plus the query as an explicit argument.
    ctx_with_query = _history_messages(history) + [
        {"role": "user", "content": query}
    ]
    history_msgs = _history_messages(history)

    # [1a] Harm check — runs before scope so a query that is both harmful and
    # out-of-scope is labeled harmful rather than merely out-of-scope.
    r["guardian_harm_score"] = backend.call_adapter(
        "guardian-core",
        ctx_with_query,
        rewriter_kwargs={
            "criteria": CRITERIA_BANK[GUARDIAN_CRITERIA],
            "scoring_schema": SCORING_SCHEMA_BANK["user_prompt"],
        },
    )["guardian"]["score"]
    if r["guardian_harm_score"] >= 0.5:
        r["blocked"] = True
        r["block_reason"] = (
            f"Harmful content detected (score={r['guardian_harm_score']:.3f})"
        )
        show_answer(r)
        return history, r

    # [1b] Scope check — is the query about government services?
    r["guardian_scope_score"] = backend.call_adapter(
        "guardian-core",
        ctx_with_query,
        rewriter_kwargs={
            "criteria": GUARDIAN_SCOPE_CRITERIA,
            "scoring_schema": SCORING_SCHEMA_BANK["user_prompt"],
        },
    )["guardian"]["score"]
    if r["guardian_scope_score"] < 0.5:
        r["blocked"] = True
        r["block_reason"] = (
            f"Out of scope - not a government services topic "
            f"(score={r['guardian_scope_score']:.3f})"
        )
        show_answer(r)
        return history, r

    # [2] Rewrite the query into a standalone form using conversation history.
    # rag.rewrite_question adds the user query to history-only context.
    rewrite = backend.call_adapter(
        "query_rewrite",
        history_msgs + [{"role": "user", "content": query}],
        num_predict=256,
    )
    parsed = rewrite.get("parsed") or {}
    r["rewritten_query"] = parsed.get("rewritten_question", query)

    # [3] Retrieve candidate documents from ChromaDB.
    retrieved = backend.retrieve(r["rewritten_query"], top_k=TOP_K)
    r["documents"] = retrieved
    mellea_docs = [
        {"doc_id": str(i), "text": t} for i, t in enumerate(retrieved)
    ]

    # [4] Answerability — can the retrieved docs answer the query?
    ans = backend.call_adapter(
        "answerability",
        history_msgs + [{"role": "user", "content": r["rewritten_query"]}],
        documents=mellea_docs,
        num_predict=8,
    )
    r["answerability"] = (ans.get("parsed") or {}).get("answerability")
    if r["answerability"] == "unanswerable":
        r["unanswerable"] = True
    else:
        # [5] Clarification — ask a follow-up if the query is still ambiguous.
        clar = backend.call_adapter(
            "query_clarification",
            history_msgs + [{"role": "user", "content": r["rewritten_query"]}],
            documents=mellea_docs,
            num_predict=256,
        )
        r["clarification"] = (clar.get("parsed") or {}).get("clarification", "")
        if not _is_clear(r["clarification"]):
            r["needs_clarification"] = True
        else:
            # [6] Answer — grounded generation from the base model (no adapter).
            prompted_query = r["rewritten_query"] + "\n\n" + GENERATION_INSTRUCTION
            r["answer"] = backend.answer(
                history_msgs + [{"role": "user", "content": prompted_query}],
                documents=mellea_docs,
                num_predict=512,
            )

            # [7] Citations — map answer spans to supporting document passages.
            if mellea_docs:
                cite = backend.call_adapter(
                    "citations",
                    ctx_with_query
                    + [{"role": "assistant", "content": r["answer"]}],
                    documents=mellea_docs,
                    num_predict=4096,
                )
                r["citations"] = cite.get("parsed") or []
            else:
                r["citations"] = []

    show_answer(r)

    # Append this turn to history. (Blocked turns returned earlier.)
    if r["unanswerable"]:
        reply = "I don't have enough information in my knowledge base to answer that."
    elif r["needs_clarification"]:
        reply = r["clarification"]
    else:
        reply = r.get("answer", "")
    history = history + [
        {"role": "user", "content": query, "documents": mellea_docs},
        {"role": "assistant", "content": reply},
    ]
    print(f"-> history now has {len(history)} message(s)")
    return history, r


# ---------------------------------------------------------------------------
# Text-mode display helpers (printing only — mirror rag_display.py)
# ---------------------------------------------------------------------------


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def show_answer(r: dict) -> None:
    """Print a single flow result. Handles all four terminal states."""
    if r.get("blocked"):
        print(f"  ⛔ BLOCKED — {r['block_reason']}")
    elif r.get("unanswerable"):
        print(f"  🔍 Not in corpus (answerability={r['answerability']})")
        print("     > I don't have enough information to answer that.")
    elif r.get("needs_clarification"):
        print("  ❓ Clarification needed:")
        print(f"     > {r['clarification']}")
    else:
        print(f"  ✅ A: {r.get('answer', '')}")


def show_intermediates(r: dict, top_k: int = TOP_K) -> None:
    """Step-by-step breakdown of a turn (text version of show_intermediates)."""
    print(f"\n  --- intermediates: {r['query']!r} ---")

    harm = r.get("guardian_harm_score", 0.0)
    print(
        f"  [1a] guardian/harm   {'🔴 harmful' if harm >= 0.5 else '🟢 safe':<12}"
        f"  score={harm:.3f}"
    )
    if r.get("blocked") and "Harmful" in r.get("block_reason", ""):
        print(f"       ⛔ {r['block_reason']}")
        return

    scope = r.get("guardian_scope_score", 0.0)
    print(
        f"  [1b] guardian/scope  "
        f"{'🟢 in-scope' if scope >= 0.5 else '🔴 out-of-scope':<12}"
        f"  score={scope:.3f}"
    )
    if r.get("blocked"):
        print(f"       ⛔ {r['block_reason']}")
        return

    print("  [2]  query_rewrite")
    print(f"       original  : {r['query']}")
    print(f"       rewritten : {r.get('rewritten_query')}")

    docs = r.get("documents", [])
    print(f"  [3]  retrieval       {len(docs)} doc(s) (top {top_k}, cosine)")

    if r.get("answerability") is not None:
        badge = "🔍 unanswerable" if r.get("unanswerable") else "✅ answerable"
        print(f"  [4]  answerability   {badge}  (verdict={r['answerability']})")
    if r.get("unanswerable"):
        return

    clar = r.get("clarification", "")
    badge = "✅ CLEAR" if _is_clear(clar) else "❓ needs clarification"
    print(f"  [5]  clarification   {badge}")
    if r.get("needs_clarification"):
        print(f"       > {clar}")
        return

    ans = r.get("answer", "")
    print(f"  [6]  answer          {len(ans)} chars")

    citations = r.get("citations", [])
    print(f"  [7]  citations       {len(citations)} found")
    for c in citations:
        rt = c.get("response_text", "").strip()
        ct = c.get("citation_text", "").strip()
        doc = c.get("citation_doc_id")
        print(f"         • {rt!r}")
        print(f"           ← doc {doc}: {ct!r}")


def show_history(history: list[dict]) -> None:
    hr("conversation history")
    if not history:
        print("  (empty)")
        return
    for m in history:
        role = "👤 user" if m["role"] == "user" else "🤖 asst"
        docs = f"  ({len(m['documents'])} docs)" if m.get("documents") else ""
        print(f"  {role}{docs}: {m['content']}")


# ---------------------------------------------------------------------------
# Demo: the notebook's 7 queries
# ---------------------------------------------------------------------------

QUERIES = [
    # Q1 — clarification: "the government service" is ambiguous (IRS vs FTB).
    "How long does it take for the government service to refund?",
    # Q2 — resolves the clarification: a 2-token reply expands via Q1 history.
    "The IRS",
    # Q3 — history-aware rewrite: "paper return instead" needs the IRS thread.
    "What if I'm filing a paper return instead?",
    # Q4 — cross-turn pronoun: "amending it" -> the paper return from Q3.
    "And what's the deadline for amending it?",
    # Q5 — rewrite + unanswerable: pronoun resolves but corpus has no fee.
    "How much does it cost?",
    # Q6 — out of scope: weather is not a government services topic.
    "What's the weather in New York tomorrow?",
    # Q7 — harm: request to forge government documents is blocked.
    "How do I forge a government ID?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print the rendered chat template sent to the model on each call",
    )
    parser.add_argument(
        "-n",
        "--num-queries",
        type=int,
        default=len(QUERIES),
        help="how many of the 7 demo queries to run (default: all)",
    )
    args = parser.parse_args()

    # Quiet mellea's INFO/WARNING chatter so the flow output is readable.
    logging.getLogger("mellea").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*TemplateRepresentation.*")

    print("Granite Switch conversational RAG flow via Ollama (Mac, no GPU)")
    backend = OllamaIntrinsicBackend(verbose=args.verbose)
    print(f"  model      : {backend.model}")
    print(f"  ollama     : {backend.ollama_url}")
    print(f"  template   : extracted from {backend.gguf_path}")
    print("  corpus     : building / loading ChromaDB (govt mt-rag subset)...")
    backend.ensure_corpus()

    history: list[dict] = []
    for i, query in enumerate(QUERIES[: args.num_queries], start=1):
        hr(f"Q{i}  {query}")
        history, r = run_conversation_turn(query, history, backend)
        show_intermediates(r)

    show_history(history)
    print("\nDone. Each adapter call routed through a control token in the prompt.")


if __name__ == "__main__":
    main()

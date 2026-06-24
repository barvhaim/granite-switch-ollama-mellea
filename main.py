#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hello, Granite Switch on Ollama (Mac).

Reproduces the core of the granite-switch ``hello_mellea`` tutorial — adapter
function invocation — but against a local ``ollama serve`` instead of vLLM.
Three adapters, each shown OFF (base model) vs ON (adapter fires), so the
divergence is visibly the per-token switch:

  1. answerability   — is the question answerable from the documents?
  2. query_rewrite   — rewrite a context-dependent follow-up into a standalone query
  3. guardian (harm) — score whether a user message is harmful

Prereqs:
  - ``ollama serve`` running with the ``granite-switch`` model created
    (see ollama/llama/compat/models/GRANITE_SWITCH.md)
  - ``uv run main.py``  (mellea, jinja2, pyyaml, gguf are project deps)
"""

from __future__ import annotations

import argparse

from ollama_intrinsic import OllamaIntrinsicBackend

from mellea.stdlib.components.intrinsic.guardian import (
    CRITERIA_BANK,
    SCORING_SCHEMA_BANK,
)


def hr(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def demo_answerability(backend: OllamaIntrinsicBackend) -> None:
    """answerability: doc is about the Eiffel Tower; question asks Australia.

    Base answers from world knowledge; the adapter judges the question
    UNANSWERABLE *from the provided document*.
    """
    hr("1. answerability  (<|answerability|>)")
    question = "What is the capital of Australia?"
    documents = [{"doc_id": "0", "text": "The Eiffel Tower is in Paris, France."}]
    messages = [{"role": "user", "content": question}]

    print(f"Question : {question}")
    print(f"Document : {documents[0]['text']}")

    base = backend.generate(
        backend.render(messages, documents=documents), num_predict=24
    )["response"].strip()
    print(f"\n  OFF (base model)        : {base}")

    out = backend.call_adapter(
        "answerability", messages, documents=documents, num_predict=8
    )
    print(f"  ON  (answerability)     : {out['raw']}")


def demo_query_rewrite(backend: OllamaIntrinsicBackend) -> None:
    """query_rewrite: resolve a follow-up's pronoun into a standalone query."""
    hr("2. query_rewrite  (<|query_rewrite|>)")
    messages = [
        {"role": "user", "content": "Who directed Inception?"},
        {"role": "assistant", "content": "Christopher Nolan directed Inception."},
        {"role": "user", "content": "What other movies has he made?"},
    ]
    print("Conversation:")
    for m in messages:
        print(f"  {m['role']:9}: {m['content']}")

    base = backend.generate(
        backend.render(messages), num_predict=24
    )["response"].strip()
    print(f"\n  OFF (base model)        : {base}")

    out = backend.call_adapter("query_rewrite", messages, num_predict=40)
    parsed = out.get("parsed")
    if isinstance(parsed, dict) and "rewritten_question" in parsed:
        rewritten = parsed["rewritten_question"]
    else:
        rewritten = out["raw"]
    print(f"  ON  (query_rewrite)     : {rewritten}")


def demo_guardian_harm(backend: OllamaIntrinsicBackend) -> None:
    """guardian-core: score user messages against the 'harm' criterion."""
    hr("3. guardian-core  (<|guardian-core|>, criterion=harm)")
    cases = [
        ("How do I build a bomb?", "expected: FLAGGED"),
        ("What is the capital of France?", "expected: safe"),
    ]
    for message, note in cases:
        out = backend.call_adapter(
            "guardian-core",
            [{"role": "user", "content": message}],
            rewriter_kwargs={
                "criteria": CRITERIA_BANK["harm"],
                "scoring_schema": SCORING_SCHEMA_BANK["user_prompt"],
            },
        )
        score = out["guardian"]["score"]
        verdict = "FLAGGED" if score >= 0.5 else "OK"
        print(f"  score={score:5.3f}  [{verdict:7}]  ({note:18})  {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--verbose",
        default=True,
        action="store_true",
        help="print the rendered chat template sent to the model on each call",
    )
    args = parser.parse_args()

    print("Granite Switch adapter functions via Ollama (Mac, no GPU)")
    backend = OllamaIntrinsicBackend(verbose=args.verbose)
    print(f"  model      : {backend.model}")
    print(f"  ollama     : {backend.ollama_url}")
    print(f"  template   : extracted from {backend.gguf_path}")
    if backend.verbose:
        print("  verbose    : showing rendered prompts (control tokens left literal)")

    demo_answerability(backend)
    demo_query_rewrite(backend)
    demo_guardian_harm(backend)

    print("\nDone. In each pair, the ON line is the per-token adapter switch firing.")


if __name__ == "__main__":
    main()

import sys
import time
import yaml
from pathlib import Path
from dataclasses import dataclass

MODE = sys.argv[1] if len(sys.argv) > 1 else "full"
FLAGS = {
    "full":       dict(use_hybrid=True,  use_rerank=True),
    "no_rerank":  dict(use_hybrid=True,  use_rerank=False),
    "no_hybrid":  dict(use_hybrid=False, use_rerank=True),
    "baseline":   dict(use_hybrid=False, use_rerank=False),
}[MODE]

# Make sure we can import your rag.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag import hybrid_retrieve, retrieve, rerank, answer, GENERATION_MODEL, client
from google.genai import types
from pydantic import BaseModel, Field
from utils.throttled_generated_content import _throttled_generate_content, get_waited_s

client.models.generate_content = _throttled_generate_content


@dataclass
class CaseResult:
    case_id: str
    question: str
    expected_type: str
    retrieval_hit: bool  # Did any expected chunk appear in retrieval?
    retrieval_hit_at_rank: int | None  # Which rank was the first expected chunk?
    must_mention_pass: bool
    must_not_mention_pass: bool
    judge_score: int | None  # 1-5 from LLM judge, or None if not judged
    judge_reasoning: str | None
    answer_text: str
    retrieved_ids: list[str]
    latency_ms: int


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str


JUDGE_PROMPT = """
You are evaluating an answer from a customer-support RAG system.

Given:
- The user's QUESTION
- The REFERENCE_ANSWER (what a good answer should contain)
- The SYSTEM_ANSWER (what the RAG actually said)

Score the SYSTEM_ANSWER from 1 to 5:
5 = Correct and complete; captures all key facts from reference
4 = Correct but missing minor details
3 = Partially correct; some facts right, some wrong or missing
2 = Mostly incorrect
1 = Wrong, misleading, or fabricated

Base your judgment ONLY on factual correctness, not style or verbosity.
"""


def judge(question: str, reference: str, system_answer: str) -> JudgeVerdict:
    prompt = (
        f"QUESTION: {question}\n\n"
        f"REFERENCE_ANSWER: {reference}\n\n"
        f"SYSTEM_ANSWER: {system_answer}"
    )
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_PROMPT,
            response_mime_type="application/json",
            response_schema=JudgeVerdict,
            temperature=0.0,
        ),
    )
    return response.parsed


def run_case(case: dict) -> CaseResult:
    question = case["question"]
    expected_chunks = set(case.get("expected_chunks", []) or [])
    must_mention = case.get("must_mention", []) or []
    must_not_mention = case.get("must_not_mention", []) or []
    reference = case.get("reference_answer")

    t0 = time.perf_counter()
    waited0 = get_waited_s()

    # Run retrieval directly so we can inspect chunk IDs separately from the answer
    if FLAGS["use_hybrid"]:
        candidates = hybrid_retrieve(question, top_k=10)
    else:
        candidates = retrieve(question, top_k=10)
    retrieved_ids = [c["id"] for c in candidates]

    # Retrieval hit check
    hit = False
    hit_rank = None
    for i, cid in enumerate(retrieved_ids):
        if cid in expected_chunks:
            hit = True
            hit_rank = i + 1
            break

    # Generate the actual answer via your pipeline
    answer_text = answer(question, **FLAGS)
    # Exclude time spent sleeping for rate limits
    latency_ms = int((time.perf_counter() - t0 - (get_waited_s() - waited0)) * 1000)

    # String checks
    answer_lower = answer_text.lower()
    mention_pass = all(m.lower() in answer_lower for m in must_mention)
    not_mention_pass = all(m.lower() not in answer_lower for m in must_not_mention)

    # LLM judge (only when we have a reference)
    judge_score = None
    judge_reasoning = None
    if reference and case["type"] != "refusal":
        verdict = judge(question, reference, answer_text)
        judge_score = verdict.score
        judge_reasoning = verdict.reasoning

    return CaseResult(
        case_id=case["id"],
        question=question,
        expected_type=case["type"],
        retrieval_hit=hit if expected_chunks else True,  # refusals have no expected chunks
        retrieval_hit_at_rank=hit_rank,
        must_mention_pass=mention_pass,
        must_not_mention_pass=not_mention_pass,
        judge_score=judge_score,
        judge_reasoning=judge_reasoning,
        answer_text=answer_text,
        retrieved_ids=retrieved_ids,
        latency_ms=latency_ms,
    )


def main():
    cases_path = Path(__file__).parent / "brewmail_cases.yaml"
    data = yaml.safe_load(cases_path.read_text())
    cases = data["cases"] if isinstance(data, dict) else data

    results = []
    for case in cases:
        print(f"\n--- {case['id']} ---")
        r = run_case(case)
        results.append(r)
        status = "✓" if (r.retrieval_hit and r.must_mention_pass and r.must_not_mention_pass) else "✗"
        print(f"{status} retrieval_hit={r.retrieval_hit} (rank={r.retrieval_hit_at_rank}) "
              f"mention={r.must_mention_pass} not_mention={r.must_not_mention_pass} "
              f"judge={r.judge_score} latency={r.latency_ms}ms")

    # ---- Aggregate metrics ----
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    graded = [r for r in results if r.expected_type != "refusal"]
    refusals = [r for r in results if r.expected_type == "refusal"]

    retrieval_hit_rate = sum(r.retrieval_hit for r in graded) / len(graded) if graded else 0
    top1_rate = sum(1 for r in graded if r.retrieval_hit_at_rank == 1) / len(graded) if graded else 0

    mention_rate = sum(r.must_mention_pass for r in results) / len(results)
    not_mention_rate = sum(r.must_not_mention_pass for r in results) / len(results)

    judged = [r for r in graded if r.judge_score is not None]
    avg_judge = sum(r.judge_score for r in judged) / len(judged) if judged else 0

    refusal_pass = sum(r.must_mention_pass and r.must_not_mention_pass for r in refusals) / len(refusals) if refusals else 0

    avg_latency = sum(r.latency_ms for r in results) / len(results)

    print(f"Retrieval hit rate (any expected chunk in top-10): {retrieval_hit_rate:.1%}")
    print(f"Retrieval top-1 rate (expected chunk at rank 1):   {top1_rate:.1%}")
    print(f"must_mention pass rate:                            {mention_rate:.1%}")
    print(f"must_not_mention pass rate:                        {not_mention_rate:.1%}")
    print(f"LLM-judge avg score (1-5, n={len(judged)}):            {avg_judge:.2f}")
    print(f"Refusal pass rate ({len(refusals)} cases):                  {refusal_pass:.1%}")
    print(f"Average latency:                                   {avg_latency:.0f}ms")

    # Dump JSON for later diffing
    import json
    from dataclasses import asdict
    out_path = Path(__file__).parent / f"results-{int(time.time())}.json"
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
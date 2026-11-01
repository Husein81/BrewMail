import os
import re
from pathlib import Path
import numpy as np
import chromadb
from google import genai
from google.genai import types
from typing import Literal
from pydantic import BaseModel
from dotenv import load_dotenv
import re
from rank_bm25 import BM25Okapi

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.1-flash-lite"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "brewmail_docs"
MIN_SIMILARITY = 0.6


class RerankedResult(BaseModel):
    chunk_id: str
    relevance_score: Literal["HIGHLY_RELEVANT", "SOMEWHAT_RELEVANT", "NOT_RELEVANT"]
    reasoning: str


class RerankResponse(BaseModel):
    ranked_results: list[RerankedResult]

RERANK_PROMPT = """
You are a re-ranker for a RAG retrieval system. You will receive a user QUESTION
and a list of CANDIDATE_CHUNKS, each with an id.

For each chunk, decide how relevant it is to the question:
- HIGHLY_RELEVANT: directly answers the question or contains the key fact needed.
- SOMEWHAT_RELEVANT: topically adjacent but not the primary answer.
- NOT_RELEVANT: on a different topic or only superficially overlapping.

Return results for every chunk id, in the ORIGINAL order you received them.
Base your judgment only on the chunk text and the question. Do not use outside knowledge.
"""

def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Re-score candidates with the LLM; return them sorted by relevance."""
    if not candidates:
        return []

    candidate_block = "\n\n".join(
        f"--- CHUNK {c['id']} ---\n"
        f"[{c['doc_type']}] {c['doc_title']} — {c['section']}\n"
        f"{c['content']}"
        for c in candidates
    )
    prompt = f"QUESTION: {query}\n\nCANDIDATE_CHUNKS:\n{candidate_block}"

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=RERANK_PROMPT,
            response_mime_type="application/json",
            response_schema=RerankResponse,
            temperature=0.0,
        ),
    )

    rank_order = {
        "HIGHLY_RELEVANT": 2,
        "SOMEWHAT_RELEVANT": 1,
        "NOT_RELEVANT": 0,
    }
    scored = []
    by_id = {c["id"]: c for c in candidates}
    for r in response.parsed.ranked_results:
        if r.chunk_id not in by_id:
            continue  # model hallucinated an id — skip
        chunk = by_id[r.chunk_id]
        scored.append({
            **chunk,
            "rerank_label": r.relevance_score,
            "rerank_rank": rank_order[r.relevance_score],
            "rerank_reasoning": r.reasoning,
        })

    # Keep only HIGHLY/SOMEWHAT; drop NOT_RELEVANT entirely.
    scored = [s for s in scored if s["rerank_rank"] > 0]
    scored.sort(key=lambda x: (-x["rerank_rank"], -x.get("rrf_score", x.get("score", 0))))  
    
    return scored
    
def chunk_markdown(filepath: str) -> list[dict]:
    text = Path(filepath).read_text(encoding="utf-8")
    title_match = re.match(r"^#\s+(.+)$", text, re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else Path(filepath).stem

    sections = re.split(r"\n(?=##\s)", text)
    chunks = []
    for i, section in enumerate(sections):
        s = section.strip()
        if s.startswith("#") and not s.startswith("##"):
            continue
        section_title = s.split("\n")[0].lstrip("#").strip() if s.startswith("##") else "Overview"
        chunks.append({
            "id": f"{Path(filepath).stem}-{i}",
            "doc_title": doc_title,
            "section": section_title,
            "content": s,
            "source_file": filepath,
        })
    return chunks


def load_all_docs(docs_dir: str = "docs") -> list[dict]:
    all_chunks = []
    for md_file in sorted(Path(docs_dir).glob("*.md")):
        all_chunks.extend(chunk_markdown(str(md_file)))
    return all_chunks


def embed(text: str) -> list[float]:
    """Embed a string. Returns a plain Python list (Chroma doesn't take numpy)."""
    response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    return list(response.embeddings[0].values)


# ------------------------------------------------------------------
# INDEX: Chroma collection, backed by ./chroma_db/
# ------------------------------------------------------------------

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},  # so distances are cosine, not L2
)

def tokenize(text: str) -> list[str]:
    """Minimal tokenization: lowercase, split on non-word chars, drop empties."""
    return [t for t in re.split(r"\W+", text.lower()) if t]

# Build the BM25 index once alongside the Chroma index.
# Store the raw corpus + chunk metadata so we can score and look up.

def build_bm25_index():
    """Pull every chunk from Chroma and build an in-memory BM25 index.
    This is cheap — a few milliseconds per 1000 chunks."""
    all_data = collection.get(include=["documents", "metadatas"])
    chunk_ids = all_data["ids"]
    chunk_texts = all_data["documents"]
    chunk_metas = all_data["metadatas"]
    # Enrich each doc text with its title and section so the BM25 index
    # can match keywords in those too (e.g. "DHL" lives only in the doc body,
    # but "Cancellation" is only in the title for some chunks).
    enriched = [
    f"{m.get('doc_title','')} {m.get('section','')} {t}"
    for m, t in zip(chunk_metas, chunk_texts)
    ]
    tokenized = [tokenize(t) for t in enriched]
    bm25 = BM25Okapi(tokenized)
    return bm25, chunk_ids, chunk_texts, chunk_metas

BM25, BM25_IDS, BM25_TEXTS, BM25_METAS = build_bm25_index()
print(f"[BM25 READY] {len(BM25_IDS)} chunks indexed for keyword search")

def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """Return the top_k chunks by BM25 score."""
    scores = BM25.get_scores(tokenize(query))
    # Attach scores to each chunk id, sort, take top_k.
    ranked = sorted(
        zip(BM25_IDS, scores, BM25_TEXTS, BM25_METAS),
        key=lambda x: x[1],
        reverse=True,
    )
    hits = []
    for chunk_id, score, text, meta in ranked[:top_k]:
        if score <= 0:  # BM25 can return 0 for no term matches
            continue
        hits.append({
            "id": chunk_id,
            "doc_title": meta.get("doc_title", ""),
            "section": meta.get("section", ""),
            "doc_type": meta.get("doc_type", "unknown"),
            "content": text,
            "bm25_score": float(score),
        })
    return hits

def reciprocal_rank_fusion(
    semantic_hits: list[dict],
    keyword_hits: list[dict],
    k: int = 60,
) -> list[dict]:
    """Combine two ranked lists by RRF. Returns hits sorted by combined rank."""
    rrf_scores = {}
    by_id = {}

    for rank, hit in enumerate(semantic_hits):
        hid = hit["id"]
        rrf_scores[hid] = rrf_scores.get(hid, 0) + 1 / (k + rank + 1)
        by_id[hid] = {**hit, "semantic_rank": rank + 1}

    for rank, hit in enumerate(keyword_hits):
        hid = hit["id"]
        rrf_scores[hid] = rrf_scores.get(hid, 0) + 1 / (k + rank + 1)
        if hid in by_id:
            by_id[hid]["keyword_rank"] = rank + 1
        else:
            by_id[hid] = {**hit, "keyword_rank": rank + 1}

    merged = []
    for hid, rrf in sorted(rrf_scores.items(), key=lambda x: -x[1]):
        chunk = by_id[hid]
        chunk["rrf_score"] = rrf
        merged.append(chunk)
    return merged


def hybrid_retrieve(
    query: str,
    top_k: int = 10,
    where: dict | None = None,
    min_similarity: float = 0.4,
) -> list[dict]:
    """Semantic + keyword, fused with RRF."""
    semantic = retrieve(query, top_k=top_k, where=where, min_similarity=min_similarity)
    keyword = bm25_search(query, top_k=top_k)

    # Metadata filter has to be applied manually to the BM25 side,
    # because rank_bm25 doesn't know about Chroma metadata.
    if where:
        keyword = [
            h for h in keyword
            if all(h.get("doc_type") == v if k == "doc_type" else True
                   for k, v in where.items())
        ]

    merged = reciprocal_rank_fusion(semantic, keyword)
    return merged[:top_k]


def classify_doc_type(doc_title: str) -> str:
    """Tiny heuristic: label chunks by what kind of doc they came from.
    In a real system this would be metadata authored upstream."""
    t = doc_title.lower()
    if "policy" in t or "cancellation" in t or "refund" in t:
        return "policy"
    if "origin" in t or "selection" in t:
        return "catalog"
    if "delivery" in t or "shipping" in t:
        return "operations"
    return "other"


def build_index():
    """Embed every chunk and store in Chroma. Idempotent — safe to re-run."""
    chunks = load_all_docs("docs")
    existing_ids = set(collection.get()["ids"])

    to_add = [c for c in chunks if c["id"] not in existing_ids]
    if not to_add:
        print(f"Index already contains {len(existing_ids)} chunks. Nothing to add.")
        return

    print(f"Embedding {len(to_add)} new chunks...")
    ids, embeddings, documents, metadatas = [], [], [], []
    for chunk in to_add:
        text_to_embed = f"{chunk['doc_title']} — {chunk['section']}\n\n{chunk['content']}"
        ids.append(chunk["id"])
        embeddings.append(embed(text_to_embed))
        documents.append(chunk["content"])
        metadatas.append({
            "doc_title": chunk["doc_title"],
            "section": chunk["section"],
            "doc_type": classify_doc_type(chunk["doc_title"]),
            "source_file": chunk["source_file"],
        })

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"Added {len(to_add)} chunks. Collection now has {collection.count()} total.")


build_index()
print(f"[INDEX READY] {collection.count()} chunks in '{COLLECTION_NAME}'")


# ------------------------------------------------------------------
# RETRIEVAL
# ------------------------------------------------------------------

def retrieve(query: str, top_k: int = 2, where: dict | None = None, min_similarity: float = MIN_SIMILARITY) -> list[dict]:
    """Chroma query, converted back to a similarity-sorted list."""
    query_vec = embed(query)
    kwargs = {"query_embeddings": [query_vec], "n_results": top_k}
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)

    # Chroma wraps every list in an outer list for batched queries. Unwrap.
    ids = results["ids"][0]
    distances = results["distances"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    hits = []
    for i in range(len(ids)):
        similarity = 1 - distances[i]  # convert cosine distance → similarity
        if similarity < min_similarity:
            continue
        hits.append({
            "id": ids[i],
            "doc_title": metadatas[i]["doc_title"],
            "section": metadatas[i]["section"],
            "doc_type": metadatas[i].get("doc_type", "unknown"),
            "content": documents[i],
            "score": similarity,
        })
    return hits


# ------------------------------------------------------------------
# GENERATION (unchanged from last lesson)
# ------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a customer support agent for BrewMail, a coffee subscription service.

Answer the user's question using ONLY the information in the DOCUMENTATION section below.
If the documentation does not contain the answer, say:
"I don't have that information. Please contact support@brewmail.com."

Do NOT use any information outside the DOCUMENTATION section.
Do NOT invent facts, policies, prices, or product details.
When you make a specific claim, cite the document title in brackets like [Refund Policy — 14-Day Window].

Keep answers concise. For factual questions, give the direct answer in 1-2 sentences.
Do not reproduce entire doc paragraphs verbatim unless the user explicitly asks for the full text.

You MAY make clear logical inferences from the documentation.
Only refuse when the docs do not contain the information needed for the inference.
"""


def answer(user_question: str, where: dict | None = None) -> str:
    candidates = hybrid_retrieve(user_question, top_k=10, where=where, min_similarity=0.4)

    print(f"\n  [HYBRID] {len(candidates)} candidates")
    for c in candidates:
        sem = c.get("semantic_rank", "-")
        kw = c.get("keyword_rank", "-")
        print(f"    rrf={c['rrf_score']:.4f}  sem#{sem} kw#{kw}  {c['doc_title']} — {c['section']}")

    if not candidates:
        return "I don't have information on that. Please contact support@brewmail.com."

    reranked = rerank(user_question, candidates)[:3]

    print(f"  [RERANKED → {len(reranked)}]")
    for r in reranked:
        print(f"    [{r['rerank_label']}]  {r['doc_title']} — {r['section']}")
        print(f"       reason: {r['rerank_reasoning']}")

    if not reranked:
        return "I don't have information on that. Please contact support@brewmail.com."

    doc_section = "\n\n".join(
        f"### {h['doc_title']} — {h['section']}\n{h['content']}" for h in reranked
    )
    prompt = f"DOCUMENTATION:\n{doc_section}\n\nUSER QUESTION: {user_question}"

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
        ),
    )
    return response.text
# ------------------------------------------------------------------
# Try it
# ------------------------------------------------------------------

questions = [
    ("How long can I pause my subscription for?", None),
    ("Do you ship coffee from Vietnam?", None),
    ("What's your refund policy?", None),
    ("What coffee origins does BrewMail ship?", None),
    ("Can I return a bag of coffee I didn't like after 30 days?", None),
    ("What's the capital of France?", None),
    ("Can I pause my subscription for 6 months?", None),
    ("What flavors of tea do you offer?", None),
    # New: metadata-filtered queries
    ("How does cancellation work?", {"doc_type": "policy"}),
    ("How does cancellation work?", {"doc_type": "catalog"}),  # should find nothing — wrong filter
    ("Where can I check the status of my DHL shipment?", None),
    ("How do I use the Pause Subscription button?", None),
]

if __name__ == "__main__":
    for q, where in questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        if where:
            print(f"   filter: {where}")
        print(f"{'='*70}")
        print(f"A: {answer(q, where=where)}")
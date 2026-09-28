from typing import List, Tuple, Dict, Any
import re

from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

from config import VECTOR_DIR, COLLECTION_NAME, EMBEDDING_MODEL, GROQ_API_KEY, GROQ_MODEL

SYSTEM_PROMPT = """You are the evidence-grounded AyurVana assistant.

Use ONLY the retrieved excerpts supplied to you from the provided Aṣṭāṅga Saṃgraha source.

Rules:
1. Do not add facts that are not supported by the retrieved excerpts.
2. Preserve the source's terminology. Do not silently replace an uncertain OCR term with a modern or corrected term.
3. If OCR appears unclear or contradictory, explicitly say that the source text/OCR is unclear and avoid guessing.
4. Distinguish what the classical text states from your own explanation. Do not present an inference as a quotation or direct statement from the text.
5. If the excerpts do not answer the question, say so clearly.
6. Do not claim that a plant cures, treats, or solves a disease unless a supplied source explicitly supports that wording; prefer 'the text describes/associates/indicates'.
7. For medical questions, do not give modern clinical advice from the classical text alone.
8. Always provide a Sources section with the page numbers used. Do not invent page numbers.
"""


def _clean_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class AyurVanaRAG:
    def __init__(self):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is missing. Put it in .env")

        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(VECTOR_DIR))
        try:
            self.collection = client.get_collection(COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"Vector collection '{COLLECTION_NAME}' was not found. "
                "Run build_index.py first."
            ) from exc

        self.groq = Groq(api_key=GROQ_API_KEY)

    def retrieve(
        self,
        question: str,
        k: int = 6,
        candidate_k: int = 18,
        max_per_page: int = 2,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """Retrieve candidates, remove duplicate text, then enforce page diversity."""
        query_embedding = self.embedder.encode(
            [question], normalize_embeddings=True
        ).tolist()[0]

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(candidate_k, k),
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        candidates = []
        seen_text = set()
        page_counts: Dict[int, int] = {}

        for doc, meta, distance in zip(docs, metas, distances):
            key = _clean_key(doc)
            if not key or key in seen_text:
                continue

            page = meta.get("page")
            if page_counts.get(page, 0) >= max_per_page:
                continue

            seen_text.add(key)
            page_counts[page] = page_counts.get(page, 0) + 1
            candidates.append((doc, meta, float(distance)))

            if len(candidates) >= k:
                break

        return candidates

    def answer(self, question: str, k: int = 6, candidate_k: int = 18):
        retrieved = self.retrieve(question, k=k, candidate_k=candidate_k)

        if not retrieved:
            return "I could not retrieve relevant source material from the indexed book.", []

        context_parts = []
        for n, (doc, meta, distance) in enumerate(retrieved, 1):
            page = meta.get("page", "unknown")
            chunk = meta.get("chunk", "unknown")
            context_parts.append(
                f"[SOURCE {n} | page {page} | chunk {chunk}]\n{doc}"
            )

        context = "\n\n".join(context_parts)

        user_prompt = f"""Question: {question}

Retrieved source excerpts:
{context}

Answer using only these excerpts. Be concise but complete. Clearly separate direct source-supported statements from explanation. If the excerpts are insufficient, say so.

At the end, include:
Sources
- page X
- page Y
Use only pages actually present above."""

        completion = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.05,
            max_completion_tokens=1800,
        )

        return completion.choices[0].message.content, retrieved

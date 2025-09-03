# pip install chromadb requests numpy
import os, json, requests, numpy as np
from typing import List
import chromadb
from chromadb import PersistentClient

# --- Config ---
CHROMA_PATH = r"/home/raregazetto/Documents/Recursive-PDF-EXTRACTION-AND-RAG/src/services/RAG/OUTPUT_DATA/chroma_db_data"         # <-- your path
COLLECTION  = "pdfs_bge_m3_cloudflare"  # <-- your collection
CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN  = os.environ["CLOUDFLARE_API_TOKEN"]

# If your documents were embedded with "passage:" prefix (typical for BGE),
# keep this True so queries use "query:" prefix for best retrieval quality.
USE_BGE_PREFIXES = True

def _normalize(vec: List[float]) -> List[float]:
    x = np.array(vec, dtype=np.float32)
    n = float(np.linalg.norm(x))
    return (x / n).tolist() if n > 0 else x.tolist()

def cf_bge_m3_embed(texts: List[str], *, input_type: str = "query") -> List[List[float]]:
    """
    Embed texts via Cloudflare Workers AI bge-m3.
    input_type: "query" for queries, "passage" for documents
    """
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/@cf/baai/bge-m3"
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    # Some BGE variants benefit from explicit prefixes
    if USE_BGE_PREFIXES:
        pref = f"{input_type}: "
        texts = [pref + t if not t.startswith(pref) else t for t in texts]

    payload = {"text": texts}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Robust parsing for common Cloudflare embeddings shapes
    result = data.get("result", {})
    arr = result.get("data") or result.get("embeddings") or result.get("vectors")

    if arr is None:
        raise RuntimeError(f"Unexpected CF response: {json.dumps(data)[:500]}")

    # CF often returns [[...]] or [{"embedding":[...]}]
    if arr and isinstance(arr[0], dict):
        embs = [item.get("embedding") or item.get("vector") for item in arr]
    else:
        embs = arr

    # L2 normalize to match cosine distance usage
    return [_normalize(v) for v in embs]

# --- Open your existing Chroma collection (no embedding_function here) ---
client = PersistentClient(path=CHROMA_PATH)
col = client.get_or_create_collection(name=COLLECTION)  # collection already has your doc vectors

def search(q: str, k: int = 5, where: dict | None = None):
    q_vec = cf_bge_m3_embed([q], input_type="query")
    res = col.query(
        query_embeddings=q_vec,
        n_results=k,
        include=["documents", "metadatas", "distances"],
        where=where or {}
    )
    # Pretty print
    for _id, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        # cosine distance -> similarity approx as (1 - distance)
        sim = 1.0 - float(dist)
        snippet = (doc or "")
        print(f"meta={meta}\n{snippet}...\n")

# ---- Example usage ----
# Narrow to a course or source if you stored metadata like {"GROUP_KEY":"EEE-EEE-313"}
search("Generate a comprehensive course outline on data communication systems.", k=20, where={"COURSE_CODE": "EEE"})
# Or without filter:
# search("Difference between MOSFETs and BJTs", k=5)

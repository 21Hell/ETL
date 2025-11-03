"""
RAG Chat FastAPI app with a minimal static UI and semantic search endpoints.

Endpoints added:
- GET  /ui                -> serves static/rag_ui.html if present
- POST /api/search        -> {text, top_k} -> semantic search via Qdrant
- GET  /api/point/{id}    -> retrieve point payload
- POST /api/chat          -> simple RAG assembly (search + naive answer)

The embedder attempts to use Deepseek when USE_DEEPSEEK=1 and the package is
available; otherwise it falls back to sentence-transformers.
"""

import os
import logging
import importlib
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
try:
    import openai
except Exception:
    openai = None

logger = logging.getLogger("rag_chat")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Chat")

# Allow local testing from any origin (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _init_qdrant_client():
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:
        logger.warning("qdrant_client not available: %s", exc)
        return None

    host = os.environ.get("QDRANT_HOST", "127.0.0.1")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    try:
        client = QdrantClient(url=f"http://{host}:{port}")
        return client
    except Exception as exc:
        logger.exception("Failed to connect to Qdrant at %s:%s: %s", host, port, exc)
        return None


_QDRANT = _init_qdrant_client()


def get_qdrant_client():
    """Return a Qdrant client, trying to reinitialize on failure.

    This helps if Qdrant was unavailable at import time but becomes available later.
    """
    global _QDRANT
    if _QDRANT is None:
        _QDRANT = _init_qdrant_client()
        return _QDRANT

    # quick health check: try a lightweight call and re-init on failure
    try:
        _QDRANT.get_collections()
        return _QDRANT
    except Exception:
        logger.info("Qdrant client appears unhealthy, attempting reinit")
        _QDRANT = _init_qdrant_client()
        return _QDRANT


def _load_embedder():
    """Load embedding model (SBERT) and optionally the Deepseek chat client.

    Returns a tuple (embed_kind, embed_model, deepseek_client).
    - embed_kind: 'sbert' when SBERT is available, else None
    - embed_model: the SBERT SentenceTransformer instance or None
    - deepseek_client: an instance of deepseek.api.DeepSeekAPI when USE_DEEPSEEK=1 and importable, else None
    """
    ds_client = None
    use_ds = os.environ.get("USE_DEEPSEEK", "false").lower() in ("1", "true", "yes")
    if use_ds:
        try:
            spec = importlib.util.find_spec("deepseek")
            if spec is not None:
                ds_mod = importlib.import_module("deepseek.api")
                # instantiate with no key; users can set DEEPSEEK_API_KEY in env if required
                DeepSeekAPI = getattr(ds_mod, "DeepSeekAPI", None)
                if DeepSeekAPI is not None:
                    try:
                        ds_client = DeepSeekAPI(api_key=os.environ.get("DEEPSEEK_API_KEY"))
                        logger.info("Loaded Deepseek chat client")
                    except Exception as exc:
                        logger.warning("Failed to instantiate DeepSeekAPI: %s", exc)
        except Exception as exc:
            logger.warning("Deepseek requested but failed to import: %s", exc)

    # Load SBERT for embeddings (preferred for vectorization)
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Using SBERT embedder 'all-MiniLM-L6-v2'")
        return "sbert", model, ds_client
    except Exception as exc:
        logger.warning("No embedder available: %s", exc)
        return None, None, ds_client


_EMBED_KIND, _EMBEDDER, _DEESEEK = _load_embedder()


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Compute embeddings for a list of texts using SBERT.

    Deepseek is a chat/LLM client and does not provide embeddings in the
    installed package; therefore SBERT is used for vectorization. If SBERT is
    not available this raises a RuntimeError.
    """
    if _EMBED_KIND == "sbert" and _EMBEDDER is not None:
        emb = _EMBEDDER.encode(texts, show_progress_bar=False)
        try:
            return emb.tolist()
        except Exception:
            return [list(e) for e in emb]

    raise RuntimeError("No embedding model available. Install sentence-transformers.")


def get_llm_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    """Try to produce a concise answer from contexts using an LLM.

    Strategy:
    1. Prefer Ollama local API/CLI if available.
    2. Fall back to Deepseek chat client if configured.
    3. Fall back to OpenAI if OPENAI_API_KEY is present.
    Raise RuntimeError if no LLM is available.
    """

    # Prepare and sanitize contexts
    ctx_texts: List[str] = []
    for c in contexts:
        t = (c.get("text") or c.get("preview") or "") if isinstance(c, dict) else str(c)
        t = t.replace('\f', ' ')
        if len(t) > 800:
            t = t[:800] + "..."
        ctx_texts.append(t)
    assembled = "\n\n".join(ctx_texts)

    # Trim very long prompts
    max_chars = 4000
    if len(assembled) > max_chars:
        assembled = assembled[:max_chars] + "\n\n[TRUNCATED]"

    prompt = (
        "Responde en español, utilizando SOLO la información en los siguientes contextos. "
        "Si la respuesta no está en los contextos, responde 'No sé'.\n\n"
        f"Contextos:\n{assembled}\n\nPregunta: {question}\n\nRespuesta concisa (máx 200 palabras):"
    )

    # 1) Ollama local API / CLI (preferred if available)
    # Default to a commonly-installed local model; allow override with OLLAMA_MODEL
    ollama_model = os.environ.get("OLLAMA_MODEL", "deepseek-r1:1.5b")
    ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    # Try HTTP API first
    try:
        import requests
        import json

        try:
            gen_url = f"{ollama_url}/api/generate"
            payload = {"model": ollama_model, "prompt": prompt, "max_tokens": 512}
            # stream the response and accumulate the 'response' field from JSON lines
            resp = requests.post(gen_url, json=payload, stream=True, timeout=60)
            if resp.ok:
                # More robust accumulation: Ollama streams may send 'thinking' deltas
                # and later emit a 'response' that can be cumulative. Keep a buffer
                # to append 'thinking' fragments and prefer the latest 'response'.
                buf = ""
                last_j = None
                for line in resp.iter_lines(decode_unicode=True, chunk_size=4096):
                    if not line:
                        continue
                    try:
                        j = json.loads(line)
                        last_j = j
                    except Exception:
                        # ignore non-json lines
                        continue
                    if not isinstance(j, dict):
                        continue
                    # 'response' is often cumulative; use it when present
                    # avoid overwriting accumulated buffer with an empty response
                    if j.get("response") is not None:
                        resp_val = str(j.get("response") or "")
                        if resp_val.strip():
                            # If the response appears cumulative (contains existing buf),
                            # replace the buffer; otherwise append incremental fragments.
                            if buf and (resp_val.startswith(buf) or buf in resp_val):
                                buf = resp_val
                            else:
                                buf += resp_val
                    # 'text' may be present as an alternative
                    elif j.get("text"):
                        # append text fragments when present
                        buf += j.get("text")
                    # 'thinking' often contains incremental tokens
                    elif j.get("thinking"):
                        buf += j.get("thinking")
                    if j.get("done"):
                        break
                if buf:
                    logger.info("Ollama HTTP returned final text (len=%d)", len(buf))
                    return buf
                # fallback to last parsed object
                if last_j and isinstance(last_j, dict):
                    for k in ("response", "text", "thinking", "content", "completion", "generated", "result"):
                        if k in last_j:
                            return last_j[k]
        except Exception as e:
            logger.exception("Ollama HTTP /api/generate request failed: %s", e)
            # try alternative endpoint
            try:
                comp_url = f"{ollama_url}/api/completions"
                payload = {"model": ollama_model, "prompt": prompt, "max_tokens": 512}
                resp = requests.post(comp_url, json=payload, timeout=10)
                if resp.ok:
                    j = resp.json()
                    if "choices" in j and j["choices"]:
                        ch = j["choices"][0]
                        if isinstance(ch, dict) and "text" in ch:
                            return ch["text"]
                        return str(ch)
            except Exception:
                logger.exception("Ollama HTTP /api/completions request failed")
                pass
    except Exception:
        # requests not available or HTTP API unreachable
        pass

    # Try Ollama CLI if HTTP failed
    try:
        import shlex
        import subprocess

        # Try 'ollama run <model> --prompt <prompt>' and 'ollama query <model> <prompt>' variants
        cmds = [
            ["ollama", "run", ollama_model, "--prompt", prompt, "--format", "json"],
            ["ollama", "query", ollama_model, prompt],
            ["ollama", "run", ollama_model, prompt],
        ]
        for cmd in cmds:
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if p.returncode == 0 and p.stdout:
                    return p.stdout.strip()
            except FileNotFoundError:
                logger.debug("Ollama CLI not found in PATH")
                break
            except Exception:
                logger.exception("Ollama CLI invocation failed: %s", cmd)
                continue
    except Exception:
        pass

    # 2) Deepseek client
    if _DEESEEK is not None:
        try:
            resp = _DEESEEK.chat_completion(prompt=prompt, prompt_sys="You are a helpful assistant", stream=False)
            if isinstance(resp, str):
                return resp
            return getattr(resp, "text", None) or getattr(resp, "content", None) or str(resp)
        except Exception as exc:
            logger.exception("Deepseek chat completion failed: %s", exc)

    # 3) OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai is not None and openai_key:
        try:
            openai.api_key = openai_key
            model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
            messages = [
                {"role": "system", "content": "You are a helpful assistant that uses given contexts to answer."},
                {"role": "user", "content": prompt},
            ]
            res = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=512, temperature=0.0)
            if res and res.choices:
                return res.choices[0].message.get("content") or str(res.choices[0])
        except Exception as exc:
            logger.exception("OpenAI call failed: %s", exc)

    raise RuntimeError("No LLM available (Ollama, Deepseek or OpenAI).")


# Serve a small UI under /ui and static files under /static
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/ui")
def ui():
    html_path = os.path.join("static", "rag_ui.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return JSONResponse({"error": "UI not found. Create static/rag_ui.html"}, status_code=404)


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


@app.get("/ready")
def ready():
    return JSONResponse({"ready": _QDRANT is not None})


@app.get("/api/llm_status")
def llm_status():
    """Return a simple status for the local Ollama service (HTTP and CLI).

    Response JSON: {available: bool, http: bool, cli: bool, models: [str], ollama_url: str}
    """
    ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    status = {"available": False, "http": False, "cli": False, "models": [], "ollama_url": ollama_url}
    # check HTTP endpoint (HEAD)
    try:
        import requests

        try:
            r = requests.head(ollama_url, timeout=1)
            status["http"] = r.status_code < 400
        except Exception:
            status["http"] = False
    except Exception:
        status["http"] = False

    # check CLI list of models
    try:
        import shutil
        import subprocess

        if shutil.which("ollama"):
            try:
                p = subprocess.run(["ollama", "ls"], capture_output=True, text=True, timeout=3)
                if p.returncode == 0 and p.stdout:
                    status["cli"] = True
                    # parse model names from lines
                    models = []
                    for ln in p.stdout.splitlines():
                        ln = ln.strip()
                        if not ln:
                            continue
                        # take first token as model name
                        models.append(ln.split()[0])
                    status["models"] = models
            except Exception:
                status["cli"] = False
    except Exception:
        status["cli"] = False

    status["available"] = bool(status.get("http") or status.get("cli"))
    return JSONResponse(status)


@app.post("/api/search")
def api_search(payload: Dict[str, Any]):
    """Search endpoint: accepts {text, top_k} and returns nearest vectors from Qdrant."""
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in request body")
    top_k = int(payload.get("top_k", 5))

    client = get_qdrant_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        cols = client.get_collections()
        collections = [c.name for c in cols.collections] if hasattr(cols, "collections") else []
    except Exception as exc:
        logger.exception("Error listing Qdrant collections: %s", exc)
        # attempt one more reinit
        client = get_qdrant_client()
        if client is None:
            raise HTTPException(status_code=500, detail="Failed to list Qdrant collections")
        try:
            cols = client.get_collections()
            collections = [c.name for c in cols.collections] if hasattr(cols, "collections") else []
        except Exception as exc2:
            logger.exception("Second attempt to list Qdrant collections failed: %s", exc2)
            raise HTTPException(status_code=500, detail="Failed to list Qdrant collections")

    if not collections:
        return JSONResponse({"results": []})

    collection = collections[0]

    # compute embedding
    try:
        vec = _embed_texts([text])[0]
    except Exception as exc:
        logger.exception("Embedding failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    try:
        hits = client.search(collection_name=collection, query_vector=vec, limit=top_k)
    except Exception as exc:
        logger.exception("Qdrant search failed: %s", exc)
        raise HTTPException(status_code=500, detail="Qdrant search failed")
    results = []
    for h in hits:
        payload = getattr(h, "payload", None) or {}
        results.append({"id": getattr(h, "id", None), "score": getattr(h, "score", None), "payload": payload})

    # Optional re-ranking using the loaded embedder (Deepseek preferred).
    try:
        if _EMBED_KIND in ("deepseek", "sbert") and _EMBEDDER is not None and results:
            # gather texts for reranking: prefer full chunk text, fallback to preview
            doc_texts = [r["payload"].get("text") or r["payload"].get("preview") or "" for r in results]
            # compute embeddings for query + docs
            emb_all = _embed_texts([text] + doc_texts)
            qv = emb_all[0]
            doc_vecs = emb_all[1:]

            # cosine similarity (pure python)
            def dot(u, v):
                return sum(a * b for a, b in zip(u, v))

            def norm(u):
                return sum(a * a for a in u) ** 0.5

            qn = norm(qv) or 1.0
            doc_norms = [norm(d) or 1.0 for d in doc_vecs]
            sims = []
            for dv, dn in zip(doc_vecs, doc_norms):
                sims.append(dot(qv, dv) / (qn * dn))

            # attach rerank score and sort results
            for r, s in zip(results, sims):
                r["rerank_score"] = float(s)

            results = sorted(results, key=lambda r: r.get("rerank_score", 0), reverse=True)
    except Exception:
        # re-ranking is best-effort; ignore failures
        pass

    return JSONResponse({"results": results, "collection": collection})


@app.get("/api/point/{point_id}")
def api_point(point_id: str):
    client = get_qdrant_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        # try to fetch point by id; qdrant-client versions differ, so prefer scroll fallback
        pt = None
        # if client exposes a get method, use it safely
        if hasattr(client, "get_point"):
            try:
                # some variants accept collection_name first, some accept only point id
                cols = client.get_collections()
                collections = [c.name for c in cols.collections] if hasattr(cols, "collections") else []
                if collections:
                    collection = collections[0]
                    try:
                        pt = client.get_point(collection_name=collection, point_id=point_id)
                    except Exception:
                        try:
                            pt = client.get_point(point_id=point_id)
                        except Exception:
                            pt = None
            except Exception:
                pt = None

        # fallback: scan the collection via scroll
        if pt is None:
            cols = client.get_collections()
            collections = [c.name for c in cols.collections] if hasattr(cols, "collections") else []
            if not collections:
                raise HTTPException(status_code=404, detail="No collections found")
            collection = collections[0]
            found = None
            for p in client.scroll(collection_name=collection, limit=1000):
                if str(getattr(p, "id", "")) == str(point_id):
                    found = p
                    break
            if not found:
                raise HTTPException(status_code=404, detail="Point not found")
            pt = found

        payload = getattr(pt, "payload", None) or {}

        # If payload lacks the original chunk text, attempt to reconstruct from local files
        if not payload.get("text") and payload.get("source") is not None and payload.get("chunk_index") is not None:
            try:
                from src.tools.ingest_books import extract_text, chunk_text
                from pathlib import Path

                src_path = Path("data/raw") / payload.get("source")
                if src_path.exists():
                    txt, pages = extract_text(src_path)
                    chunks = list(chunk_text(txt))
                    ci = int(payload.get("chunk_index"))
                    if 0 <= ci < len(chunks):
                        ch = chunks[ci]
                        # compute positions to estimate page
                        search_pos = 0
                        positions = []
                        for c in chunks:
                            idx = txt.find(c, search_pos)
                            if idx == -1:
                                idx = search_pos
                            positions.append(idx)
                            search_pos = idx + len(c)

                        start = positions[ci]
                        mid = start + max(0, len(ch) // 2)
                        page_no = None
                        if pages:
                            page_offsets = []
                            pos = 0
                            for p in pages:
                                page_offsets.append(pos)
                                pos += len(p) + 2
                            for pi in range(len(page_offsets)):
                                start_off = page_offsets[pi]
                                end_off = page_offsets[pi + 1] if pi + 1 < len(page_offsets) else (page_offsets[-1] + len(pages[-1]))
                                if mid >= start_off and mid <= end_off:
                                    page_no = pi + 1
                                    break

                        payload["text"] = ch
                        payload["page"] = page_no
            except Exception:
                # best-effort only; don't fail the request
                pass

        return JSONResponse({"id": getattr(pt, "id", None), "payload": payload})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch point %s: %s", point_id, exc)
        raise HTTPException(status_code=404, detail="Point not found")


@app.post("/api/chat")
def api_chat(body: Dict[str, Any]):
    """Simple RAG chat: perform semantic search and return the top contexts + a naive assembled answer."""
    text = body.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in request body")
    top_k = int(body.get("top_k", 5))

    # reuse search
    search_resp = api_search({"text": text, "top_k": top_k})
    data = search_resp.body if hasattr(search_resp, "body") else search_resp
    # Depending on FastAPI internals, data may be a Response; normalize
    if isinstance(data, bytes):
        import json as _json

        data = _json.loads(data)

    results = data.get("results", []) if isinstance(data, dict) else []

    # extract payloads and deduplicate by text/preview to avoid duplicate chunks
    contexts: List[Dict[str, Any]] = []
    seen_texts = set()
    for r in results:
        p = r.get("payload") or {}
        key = (p.get("text") or p.get("preview") or "").strip()
        if not key:
            continue
        if key in seen_texts:
            continue
        seen_texts.add(key)
        contexts.append(p)
        if len(contexts) >= top_k:
            break

    # Try to generate an LLM answer (Deepseek -> OpenAI). If that fails, fall back
    # to returning the assembled contexts with a note.
    try:
        answer_text = get_llm_answer(text, contexts)
        return JSONResponse({"answer": answer_text, "contexts": contexts, "raw_results": results})
    except Exception as exc:
        logger.info("LLM synthesis unavailable or failed: %s", exc)

    # Fallback: naive assembly
    assembled_list = [ (c.get("text") or c.get("preview") or "") if isinstance(c, dict) else str(c) for c in contexts ]
    assembled = "\n\n".join(assembled_list)
    answer = f"Retrieved {len(contexts)} contexts.\n\n{assembled}\n\n(Use an LLM to generate a fluent answer from these contexts.)"
    return JSONResponse({"answer": answer, "contexts": contexts, "raw_results": results})

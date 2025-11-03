"""ingest_books.py

Minimal script to extract text from local books in `data/raw`, chunk them,
optionally compute embeddings (if `sentence-transformers` is installed), and
upload points to Qdrant (if `qdrant-client` installed).

Usage:
  python -m src.tools.ingest_books --dry-run
  python -m src.tools.ingest_books --collection my_books

This script is defensive: it will run in dry-run mode without external deps and
print what it *would* do. Install dependencies for full behavior:
  pip install qdrant-client==1.11.0 sentence-transformers ebooklib pdfminer.six tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Iterable, List, Optional

DATA_RAW = Path("data/raw")


def list_books() -> List[Path]:
    if not DATA_RAW.exists():
        return []
    return [p for p in DATA_RAW.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".epub", ".txt"}]


def extract_text(path: Path) -> str:
    """Extract text from pdf/epub/txt with best-effort fallbacks.

    Does not require external libs for metadata-only/dry-run; if libraries are
    missing it returns an informative placeholder string.
    """
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".epub":
        try:
            from ebooklib import epub
            from bs4 import BeautifulSoup

            book = epub.read_epub(str(path))
            items = []
            for item in book.get_items():
                if item.get_type() == epub.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    items.append(soup.get_text(separator=" "))
            # Return full text and pages=[] (no page concept in epub)
            return "\n\n".join(items), []
        except Exception:
            return f"[EPUB] Could not extract text for {path.name} (missing libs or parse error)", []

    if suffix == ".pdf":
        try:
            # extract per-page text when possible so we can reference page numbers
            from pdfminer.high_level import extract_text as pdf_extract
            from pdfminer.pdfpage import PDFPage

            # count pages
            with open(str(path), "rb") as fh:
                pages_count = sum(1 for _ in PDFPage.get_pages(fh))

            pages = []
            for i in range(pages_count):
                try:
                    ptext = pdf_extract(str(path), page_numbers=[i])
                except Exception:
                    ptext = ""
                pages.append(ptext)

            full = "\n\n".join(pages)
            return full, pages
        except Exception:
            return f"[PDF] Could not extract text for {path.name} (missing libs or parse error)", []

    return f"[UNKNOWN] Skipping unsupported file {path.name}", []


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> Iterable[str]:
    words = text.split()
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        yield " ".join(chunk)
        i += chunk_size - overlap


def compute_embeddings(texts: List[str]):
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts, show_progress_bar=True)
    except Exception as exc:
        raise RuntimeError("sentence-transformers not installed or failed to run: " + str(exc))


def upload_to_qdrant(collection_name: str, vectors, payloads, host: str = "127.0.0.1", port: int = 6333):
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import VectorParams
    except Exception as exc:
        raise RuntimeError("qdrant-client not installed: " + str(exc))

    client = QdrantClient(url=f"http://{host}:{port}")

    # Ensure vectors are plain python lists (avoid numpy truthiness issues)
    vectors_list = []
    for v in vectors:
        if hasattr(v, "tolist"):
            vectors_list.append(v.tolist())
        else:
            # cast to list in case it's a tuple or other sequence
            vectors_list.append(list(v))

    # create collection if not exists (vector size from first vector)
    if not vectors_list:
        raise RuntimeError("No vectors to upload")

    dim = len(vectors_list[0])
    # If collection doesn't exist, create it. Do NOT recreate (destructive).
    try:
        existing = client.get_collection(collection_name=collection_name)
        # If vector size mismatches, we still attempt to proceed but warn.
        existing_size = existing.vectors.config.size if getattr(existing, "vectors", None) is not None else None
        if existing_size is not None and existing_size != dim:
            print(f"Warning: existing collection vector size={existing_size} differs from computed dim={dim}")
    except Exception:
        # collection missing: create it
        try:
            client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=dim, distance="Cosine"))
            print(f"Created collection {collection_name} with dim={dim}")
        except Exception as exc:
            # if creation fails, re-raise
            raise

    # Build points using deterministic string ids to allow incremental upserts.
    # ID format: "{source}::{chunk_index}" so re-running for the same file will update
    # the same points instead of duplicating or requiring reassigning numeric ids.
    points = []
    for vec, payload in zip(vectors_list, payloads):
        src = payload.get("source", "unknown")
        idx = payload.get("chunk_index", 0)
        # create a deterministic UUID based on source and chunk index so re-running
        # the ingestion for the same book/chunk updates the same point
        point_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"{src}::{idx}")
        points.append({"id": str(point_uuid), "vector": vec, "payload": payload})

    # upload in batch (qdrant-client will create or update points with the given ids)
    client.upsert(collection_name=collection_name, points=points)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Don't upload, just show actions")
    p.add_argument("--collection", type=str, default="books_collection", help="Qdrant collection name to use")
    p.add_argument("--top-dir", type=str, default="data/raw", help="Directory with books")
    p.add_argument("--file", type=str, help="Process a single file (path) inside --top-dir or absolute path")
    args = p.parse_args()

    top = Path(args.top_dir)
    if args.file:
        fpath = Path(args.file)
        # if relative path, resolve relative to top-dir
        if not fpath.is_absolute():
            fpath = top / fpath
        if not fpath.exists() or not fpath.is_file():
            print("Specified file not found:", fpath)
            return
        files = [fpath]
    else:
        files = list_books() if top == DATA_RAW else [p for p in Path(args.top_dir).iterdir() if p.is_file()]

    if not files:
        print("No books found in", top)
        return

    print(f"Found {len(files)} book(s) in {top}")

    all_texts = []
    metadata = []
    for f in files:
        print("-", f.name)
        text, pages = extract_text(f)
        # keep a short preview for payload
        preview = text[:400].strip().replace("\n", " ") if isinstance(text, str) else ""

        chunks = list(chunk_text(text))

        # compute character positions for each chunk to estimate page
        search_pos = 0
        positions = []
        for ch in chunks:
            idx = text.find(ch, search_pos)
            if idx == -1:
                idx = search_pos
            positions.append(idx)
            search_pos = idx + len(ch)

        # build page offsets
        page_offsets = []
        if pages:
            pos = 0
            for p in pages:
                page_offsets.append(pos)
                pos += len(p) + 2

        for i, ch in enumerate(chunks):
            # determine page by mid-point of chunk
            start = positions[i]
            mid = start + max(0, len(ch) // 2)
            page_no = None
            if pages:
                for pi in range(len(page_offsets)):
                    start_off = page_offsets[pi]
                    end_off = page_offsets[pi + 1] if pi + 1 < len(page_offsets) else (page_offsets[-1] + len(pages[-1]))
                    if mid >= start_off and mid <= end_off:
                        page_no = pi + 1
                        break

            all_texts.append(ch)
            metadata.append({
                "source": f.name,
                "chunk_index": i,
                "preview": preview,
                "text": ch,
                "page": page_no,
            })

    print(f"Total chunks generated: {len(all_texts)}")

    if args.dry_run:
        print("Dry-run mode: not computing embeddings or uploading. Sample payloads:")
        for i, md in enumerate(metadata[:5]):
            print(i, json.dumps(md, ensure_ascii=False))
        return

    # compute embeddings
    print("Computing embeddings (requires sentence-transformers)...")
    try:
        vectors = compute_embeddings(all_texts)
    except Exception as exc:
        print("Failed to compute embeddings:", exc)
        return

    print("Uploading to Qdrant (requires qdrant-client)...")
    try:
        upload_to_qdrant(args.collection, vectors, metadata)
    except Exception as exc:
        print("Failed to upload to Qdrant:", exc)
        return

    print("Done: uploaded", len(vectors), "vectors to collection", args.collection)


if __name__ == "__main__":
    main()

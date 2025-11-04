import os
import re
import sqlite3
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# Setup
# =========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_PATH = "kb.db"
EMBED_MODEL = "text-embedding-3-small"

BASE_FOLDER = "knowledge_base"
CATEGORIES = {
    "ikhtiar_hamil": "ikhtiar_hamil.txt",
    "sedang_hamil":  "sedang_hamil.txt",
    "lain_lain":     "lain_lain.txt",
}

# Q/A parser: one block = Q: ... \n A: ... (until next Q: or EOF)
QA_BLOCK = re.compile(
    r"^\s*Q:\s*(.*?)\s*\n\s*A:\s*(.*?)(?=\n\s*Q:|\Z)",
    re.S | re.M
)

# =========================
# DB utils
# =========================
def connect_db():
    return sqlite3.connect(DB_PATH)

def reset_schema(conn):
    cur = conn.cursor()
    # Fresh QA table (drop & recreate to avoid stale rows)
    cur.execute("DROP TABLE IF EXISTS knowledge_base_qa")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base_qa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category    TEXT NOT NULL,
            question    TEXT NOT NULL,
            answer      TEXT NOT NULL,
            q_embedding BLOB NOT NULL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_kbqa_cat ON knowledge_base_qa(category)")
    conn.commit()

# =========================
# Embeddings
# =========================
def embed_batch(texts):
    """Return list[np.ndarray(float32)] for a list of strings."""
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]

# =========================
# Text cleaning helpers
# =========================
def _strip_label_prefix(s: str, label: str):
    """
    Remove leading 'Q:' or 'A:' (with optional spaces), case-insensitive.
    Example: 'A:  Hello' → 'Hello'
    """
    s2 = s.lstrip()
    prefix = f"{label}:"
    if s2.lower().startswith(prefix.lower()):
        s2 = s2[len(prefix):].lstrip()
    return s2

def _normalize_ws(s: str):
    # collapse weird spacing, keep newlines as-is
    # but within lines, compress multiple spaces
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in s.splitlines()]
    # keep blank lines between paragraphs
    return "\n".join([ln for ln in lines if ln != "" or True])

# =========================
# Parsing
# =========================
def parse_qa_file(path):
    """Return list of (question, answer) pairs from a Q/A text file."""
    raw = open(path, "r", encoding="utf-8").read()
    pairs = []
    for m in QA_BLOCK.finditer(raw):
        q = m.group(1).strip()
        a = m.group(2).strip()

        # Defensive cleanup: strip any lingering labels + normalize whitespace
        q = _strip_label_prefix(q, "Q")
        a = _strip_label_prefix(a, "A")
        q = _normalize_ws(q)
        a = _normalize_ws(a)

        if q and a:
            pairs.append((q, a))
    return pairs

# =========================
# Ingestion
# =========================
def ingest_category(conn, category, filepath, batch_size=40):
    cur = conn.cursor()

    if not os.path.exists(filepath):
        print(f"⚠️  File not found: {filepath} (skipping)")
        return 0

    pairs = parse_qa_file(filepath)
    if not pairs:
        print(f"⚠️  No Q/A blocks found in {filepath}. Ensure 'Q:' then 'A:' format.")
        return 0

    # dedupe against existing rows (by exact question text)
    cur.execute("SELECT question FROM knowledge_base_qa WHERE category=?", (category,))
    existing_qs = {row[0] for row in cur.fetchall()}
    pairs = [(q, a) for (q, a) in pairs if q not in existing_qs]

    if not pairs:
        print(f"ℹ️  Nothing new to insert for {category}.")
        return 0

    # batch embed questions
    inserted = 0
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        qs = [q for q, _ in batch]
        vecs = embed_batch(qs)

        for (q, a), emb in zip(batch, vecs):
            cur.execute(
                "INSERT INTO knowledge_base_qa (category, question, answer, q_embedding) VALUES (?, ?, ?, ?)",
                (category, q, a, emb.tobytes())
            )
            inserted += 1

        conn.commit()

    print(f"✅ {category}: inserted {inserted} Q/A rows from {os.path.basename(filepath)}")
    return inserted

def ingest_all():
    conn = connect_db()
    reset_schema(conn)

    total = 0
    for cat, fname in CATEGORIES.items():
        path = os.path.join(BASE_FOLDER, fname)
        print(f"➡️  Ingesting {path} as '{cat}'")
        total += ingest_category(conn, cat, path)

    conn.close()
    print(f"\n🎉 Done. Total inserted: {total}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    ingest_all()

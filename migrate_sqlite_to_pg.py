import os
import sqlite3
import psycopg
from psycopg.rows import dict_row

# =========================
# CONFIG
# =========================

# Lokasi SQLite lama (fail kb.db)
SQLITE_PATH = os.getenv("SQLITE_PATH", "kb.db")

# Postgres URL – pakai External Database URL dari Render
# Tip: untuk senang, set dalam environment: DATABASE_URL=...
PG_URL = os.getenv("DATABASE_URL") or "postgresql://efferty_chatbot_db_user:password@host:5432/efferty_chatbot_db"

# =========================
# HELPER
# =========================

def connect_sqlite(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"SQLite file not found: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def connect_postgres(url: str):
    if not url.startswith("postgresql://"):
        raise SystemExit("DATABASE_URL mesti bermula dengan 'postgresql://...'")
    conn = psycopg.connect(url, row_factory=dict_row)
    conn.autocommit = False
    return conn

def ensure_tables_pg(pg):
    cur = pg.cursor()
    # sama macam dalam connection.py (versi Postgres)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base_qa (
            id SERIAL PRIMARY KEY,
            category TEXT,
            question TEXT,
            answer   TEXT,
            q_embedding BYTEA
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS generated_answers (
            id SERIAL PRIMARY KEY,
            category TEXT,
            user_question TEXT,
            ai_answer TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            approved INTEGER DEFAULT 0
        )
    """)
    pg.commit()
    cur.close()

# =========================
# MIGRATE KB QA
# =========================

def migrate_kb(sqlite_conn, pg_conn):
    s_cur = sqlite_conn.cursor()
    p_cur = pg_conn.cursor()

    # check exist
    try:
        s_cur.execute("SELECT category, question, answer, q_embedding FROM knowledge_base_qa")
    except sqlite3.OperationalError:
        print("[KB] Table knowledge_base_qa tak wujud dalam SQLite. Skip.")
        return

    rows = s_cur.fetchall()
    print(f"[KB] Jumlah row dari SQLite: {len(rows)}")

    if not rows:
        return

    # optional: kosongkan dulu table di Postgres kalau nak fresh
    # p_cur.execute("TRUNCATE TABLE knowledge_base_qa RESTART IDENTITY")

    insert_sql = """
        INSERT INTO knowledge_base_qa (category, question, answer, q_embedding)
        VALUES (%s, %s, %s, %s)
    """

    batch = []
    for r in rows:
        batch.append((
            r["category"],
            r["question"],
            r["answer"],
            r["q_embedding"],
        ))

    p_cur.executemany(insert_sql, batch)
    pg_conn.commit()
    p_cur.close()
    print(f"[KB] Berjaya import {len(rows)} row ke Postgres.")

# =========================
# MIGRATE GENERATED ANSWERS
# =========================

def migrate_generated(sqlite_conn, pg_conn):
    s_cur = sqlite_conn.cursor()
    p_cur = pg_conn.cursor()

    try:
        s_cur.execute("""
            SELECT category, user_question, ai_answer, created_at, approved
            FROM generated_answers
        """)
    except sqlite3.OperationalError:
        print("[GEN] Table generated_answers tak wujud dalam SQLite. Skip.")
        return

    rows = s_cur.fetchall()
    print(f"[GEN] Jumlah row dari SQLite: {len(rows)}")

    if not rows:
        return

    # optional: kosongkan dulu table di Postgres kalau nak fresh
    # p_cur.execute("TRUNCATE TABLE generated_answers RESTART IDENTITY")

    insert_sql = """
        INSERT INTO generated_answers (category, user_question, ai_answer, created_at, approved)
        VALUES (%s, %s, %s, %s, %s)
    """

    batch = []
    for r in rows:
        batch.append((
            r["category"],
            r["user_question"],
            r["ai_answer"],
            r["created_at"],
            r["approved"],
        ))

    p_cur.executemany(insert_sql, batch)
    pg_conn.commit()
    p_cur.close()
    print(f"[GEN] Berjaya import {len(rows)} row ke Postgres.")

# =========================
# MAIN
# =========================

def main():
    print("=== MIGRATE SQLite → Postgres ===")
    print("SQLite path :", SQLITE_PATH)
    print("PG URL set? :", bool(PG_URL))

    sqlite_conn = connect_sqlite(SQLITE_PATH)
    pg_conn = connect_postgres(PG_URL)

    ensure_tables_pg(pg_conn)

    migrate_kb(sqlite_conn, pg_conn)
    migrate_generated(sqlite_conn, pg_conn)

    sqlite_conn.close()
    pg_conn.close()
    print("=== SIAP. Semua migrate (kalau ada data) ✨ ===")

if __name__ == "__main__":
    main()

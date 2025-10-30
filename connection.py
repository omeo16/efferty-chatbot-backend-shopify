import os
import re
import sqlite3
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from langdetect import detect
from deep_translator import GoogleTranslator

# =========================
# Config
# =========================
load_dotenv(find_dotenv(), override=True)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise Exception("Set OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_KEY)

# Models
EMBED_MODEL = "text-embedding-3-small"   
CHAT_MODEL  = "gpt-4o-mini"

DB_PATH = "kb.db"

TOP_N = 8
SIM_THRESHOLD = 0.10   
KW_THRESHOLD  = 0.08

COS_MIN       = 0.20     # minimum cosine to be considered "plausible"
KW_MIN        = 0.12     # minimum keyword overlap to help a low-ish cosine
SELECTED_MIN  = 0.24     # if selected-category top score below this, allow switching
RESCUE_DELTA  = 0.06     # switch if global best beats selected by this margin

# Debug
DEBUG_CANDIDATES = os.getenv("DEBUG_CANDIDATES", "false").lower() == "true"

# Strict mode (if true, never use GPT reasoning fallback)
STRICT_MODE = os.getenv("STRICT_MODE", "false").lower() == "true"
print(">>> DEBUG: STRICT_MODE from .env =", os.getenv("STRICT_MODE"))
print(">>> DEBUG: parsed STRICT_MODE =", STRICT_MODE)

# Categories
CATEGORIES = ["ikhtiar_hamil", "sedang_hamil", "lain_lain"]

# Files sync
BASE_FOLDER = "knowledge_base"
CATEGORY_FILES = {
    "ikhtiar_hamil": "ikhtiar_hamil.txt",
    "sedang_hamil":  "sedang_hamil.txt",
    "lain_lain":     "lain_lain.txt",
}

# Platform URLs
TIKTOK_URL  = os.getenv("TIKTOK_URL",  "https://www.tiktok.com/@efferty?is_from_webapp=1&sender_device=pc")
SHOPEE_URL  = os.getenv("SHOPEE_URL",  "https://shopee.com.my/efferty")
WEBSITE_URL = os.getenv("WEBSITE_URL", "https://deals.efferty.com")
WHATSAPP_URL = os.getenv("WHATSAPP_URL" ,"https://wa.me/601126259641?text=Hi%2C+saya+ingin+mengetahui+lebih+lanjut+mengenai+Efferty.")

# Smart justification
JUSTIFY_MODE = os.getenv("JUSTIFY_MODE", "true").lower() == "true"

SMART_DISCLAIMER = os.getenv(
    "SMART_DISCLAIMER",
    "Disclaimer: Sesetengah informasi kemungkinan tidak tepat, sila hubungi kumpulan kami untuk maklumat lanjut. Sila tekan butang 4."
)

app = Flask(__name__)
CORS(app)

# =========================
# Small utils
# =========================
def _escape_regex(s: str) -> str:
    return re.escape(s)

def linkify_platforms(text: str) -> str:
    if not text or "<a " in text.lower():
        return text
    patterns = [
        (["tiktok shop", "tiktok"], TIKTOK_URL),
        (["shopee"], SHOPEE_URL),
        (["website rasmi kami", "laman web rasmi", "website rasmi", "website", "laman web"], WEBSITE_URL),
        (["whatsapp"], WHATSAPP_URL),
    ]
    html = text
    for labels, url in patterns:
        labels_sorted = sorted(labels, key=len, reverse=True)
        alt = "|".join(_escape_regex(x) for x in labels_sorted)
        regex = re.compile(rf"(?<![\w/])({alt})(?![^<]*>)", re.IGNORECASE)
        def _repl(m):
            label = m.group(1)
            return f'<a href="{url}" target="_blank" rel="noopener">{label}</a>'
        html = regex.sub(_repl, html)
    return html

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- Category normalization (handles 'Sedang Hamil  ' vs 'sedang_hamil') ---
def _norm_cat_key(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")

# --- Lightweight BM/EN synonym expansion (helps short/slang queries) ---
BM_SYNONYMS = {
    "macam mana": ["bagaimana", "mcm mana", "camna", "cara"],
    "consume": ["minum", "ambil", "pengambilan", "guna", "consume"],
    "susu": ["susu efferty", "efferty"],
    "kenapa": ["mengapa", "sebab", "justifikasi"],
}
def _expand_query(q: str) -> str:
    ql = (q or "").lower()
    extra = []
    for key, alts in BM_SYNONYMS.items():
        if key in ql:
            extra.extend(alts)
    return q if not extra else f"{q} " + " ".join(sorted(set(extra)))

def _load_qa_rows(category: str):
    conn = _connect()
    c = conn.cursor()
    try:
        # TRIM + normalize to avoid whitespace/label mismatches
        c.execute("""
            SELECT id, question, answer, q_embedding
            FROM knowledge_base_qa
            WHERE lower(replace(trim(category), ' ', '_')) = ?
        """, (_norm_cat_key(category),))
        rows = c.fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    if DEBUG_CANDIDATES:
        print(f"[DEBUG] loaded rows for {category} =", len(rows))
    items = []
    for row in rows:
        emb_blob = row["q_embedding"]
        if emb_blob is None:
            continue
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        items.append({"id": row["id"], "q": row["question"], "a": row["answer"], "emb": emb})
    return items

def _embed(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.asarray(resp.data[0].embedding, dtype=np.float32)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _tokens(s: str):
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def _kw_overlap(query: str, text: str) -> float:
    A, B = set(_tokens(query)), set(_tokens(text))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def _clean_prefixes(s: str) -> str:
    t = s.lstrip()
    t = re.sub(r"^(?:[-*\s]*)(?:a:|answer:)\s*", "", t, flags=re.I)
    t = re.sub(r"^\s*q:\s*.*\n+", "", t, flags=re.I)
    return t.strip()

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

# intent & shape detectors
def _looks_like_why(q: str) -> bool:
    ql = (q or "").lower()
    return any(w in ql for w in ["kenapa", "mengapa", "sebab", "why", "justifikasi", "justify"])

def _looks_like_how(q: str) -> bool:
    ql = (q or "").lower()
    return any(w in ql for w in ["bagaimana", "macam mana", "how"])

def _looks_like_catalog(answer: str) -> bool:
    a = (answer or "").lower().strip()
    has_list_markers = bool(re.search(r"\b(\d+\.\s|1\)|•|- )", a))
    starts_with_terdapat = a.startswith("terdapat ")
    many_commas = a.count(",") >= 3
    keywords = any(k in a for k in ["kategori", "perisa", "pilihan", "signature", "essential", "premium"])
    return has_list_markers or starts_with_terdapat or (many_commas and keywords)

# =========================
# Retrieval & re-ranking
# =========================
def retrieve_candidates(question: str, category: str, top_n: int = TOP_N):
    rows = _load_qa_rows(category)
    if not rows:
        return []
    qvec = _embed(_expand_query(question))
    scored = []
    for r in rows:
        cos = _cosine(qvec, r["emb"])
        scored.append({**r, "cos": cos})
    scored.sort(key=lambda x: x["cos"], reverse=True)
    cand = scored[:max(1, top_n)]
    short_query = len(question.split()) <= 6
    cos_w, kw_w = (0.6, 0.4) if short_query else (0.7, 0.3)
    for r in cand:
        kw = _kw_overlap(question, r["q"])
        r["kw"] = kw
        r["score"] = cos_w * r["cos"] + kw_w * r["kw"]
    cand.sort(key=lambda x: x["score"], reverse=True)

    if DEBUG_CANDIDATES:
        print("[DEBUG] top candidates:")
        for r in cand[:5]:
            print(f"  score={r['score']:.3f} cos={r['cos']:.3f} kw={r['kw']:.3f}  Q={r['q'][:80]}")
    return cand

def retrieve_candidates_any(question: str, top_n: int = TOP_N):
    best = None
    best_cat = None
    for cat in CATEGORIES:
        cands = retrieve_candidates(question, cat, top_n=top_n)
        if not cands:
            continue
        top = cands[0]
        if (best is None) or (top["score"] > best["score"]):
            best = top
            best_cat = cat
    return best, best_cat

def related_kb_questions(question: str, category: str, exclude_q: str = None, n: int = 3):
    conn = _connect()
    c = conn.cursor()
    try:
        c.execute("""
            SELECT id, question, q_embedding
            FROM knowledge_base_qa
            WHERE lower(replace(trim(category), ' ', '_')) = ?
        """, (_norm_cat_key(category),))
        rows = c.fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    if not rows:
        return []
    qvec = _embed(_expand_query(question))
    scored = []
    for row in rows:
        q_text = row["question"]
        if exclude_q and q_text.strip().lower() == exclude_q.strip().lower():
            continue
        emb = np.frombuffer(row["q_embedding"], dtype=np.float32)
        cos = _cosine(qvec, emb)
        scored.append((cos, q_text))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [q for _, q in scored[:n]]

# =========================
# Grounded justification helpers
# =========================
def _gather_fact_context(category: str, keywords, limit_pairs: int = 6, max_chars: int = 1800) -> str:
    conn = _connect()
    c = conn.cursor()
    if not keywords:
        return ""
    kw_like = " OR ".join(["question LIKE ? OR answer LIKE ?"] * len(keywords))
    params = []
    for kw in keywords:
        like = f"%{kw}%"
        params.extend([like, like])
    try:
        c.execute(
            f"""
            SELECT question, answer FROM knowledge_base_qa
            WHERE lower(replace(trim(category), ' ', '_')) = ? AND ({kw_like})
            ORDER BY id DESC LIMIT ?
            """,
            (_norm_cat_key(category), *params, limit_pairs)
        )
        rows = c.fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()
    chunks = []
    total = 0
    for r in rows:
        block = f"Q: {r['question']}\nA: {r['answer']}\n"
        total += len(block)
        if total > max_chars:
            break
        chunks.append(block)
    return "\n".join(chunks).strip()

def _need_explanation(user_q: str, kb_answer: str) -> bool:
    q = _normalize(user_q)
    a = _normalize(kb_answer)
    trigger_words = ["kenapa", "mengapa", "sebab", "why", "justifikasi", "justify"]
    cond_words    = ["pcos", "endometriosis", "fibroid", "cyst", "haid", "period"]
    ask_why = any(w in q for w in trigger_words)
    mention_cond = any(w in q for w in cond_words)
    looks_short = len(a) <= 90
    return (ask_why or mention_cond) and looks_short

def _justify_answer(user_q: str, kb_answer: str, category: str) -> str:
    base_tokens = re.findall(r"[a-zA-Z0-9]+", (user_q + " " + kb_answer), flags=re.I)
    extra = ["efferty", "susu", "cinnamon", "coklat", "pcos", "sesuai", "kandungan", "cara", "pengambilan"]
    keywords = list(dict.fromkeys([t.lower() for t in base_tokens + extra if len(t) >= 3]))
    context = _gather_fact_context(category, keywords)
    if not context:
        return ""
    system_prompt = (
        "You are EffertyAskMe. Provide a SHORT explanation in the user's language "
        "based ONLY on the provided Knowledge Base facts.\n"
        "Rules:\n"
        "- Do not invent medical claims or guarantee outcomes.\n"
        "- Do not diagnose or prescribe. Keep it general and practical.\n"
        "- Keep it to 1–3 sentences max.\n"
        "- If the KB facts are insufficient to justify, reply with an empty string.\n"
        "- End with: 'Nasihat umum sahaja; dapatkan nasihat doktor jika ragu.'"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Knowledge Base facts:\n{context}"},
        {"role": "user", "content": f"Soalan: {user_q}\nJawapan KB (ringkas): {kb_answer}\nTerangkan sebab/rasional secara ringkas, berpandukan fakta KB di atas."}
    ]
    try:
        chat = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, temperature=0.2, max_tokens=160
        )
        expl = _clean_prefixes((chat.choices[0].message.content or "").strip())
        if len(_normalize(expl)) < 12:
            return ""
        return expl
    except Exception as e:
        print("Justifier error:", e)
        return ""

# =========================
# TXT sync + admin learning table
# =========================
def _qa_pairs_for_category(category: str):
    conn = _connect()
    c = conn.cursor()
    try:
        c.execute("""SELECT question, answer FROM knowledge_base_qa WHERE category = ? ORDER BY id ASC""", (category,))
        rows = c.fetchall()
    finally:
        conn.close()
    return [(r["question"], r["answer"]) for r in rows]

def _sync_category_txt(category: str):
    fname = CATEGORY_FILES.get(category)
    if not fname:
        print(f"[SYNC] No txt mapping for category: {category}")
        return
    import os
    os.makedirs(BASE_FOLDER, exist_ok=True)
    path = os.path.join(BASE_FOLDER, fname)
    pairs = _qa_pairs_for_category(category)
    lines = []
    for q, a in pairs:
        q_clean = (q or "").strip()
        a_clean = (a or "").rstrip()
        lines.append(f"Q: {q_clean}\nA:\n{a_clean}\n")
    contents = "\n".join(lines).rstrip() + "\n" if lines else ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(contents)
    print(f"[SYNC] Wrote {len(pairs)} Q/A -> {path}")

def _ensure_generated_table():
    try:
        conn = _connect()
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS generated_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                user_question TEXT,
                ai_answer TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                approved INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print("generated_answers init error:", e)
_ensure_generated_table()

# =========================
# SMART /ask endpoint
# =========================
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True)
        category = data.get("category")
        question  = (data.get("question") or "").strip()

        # convo memory (optional from frontend)
        history       = data.get("history") or []   # [{role, content}]
        prev_question = (data.get("prev_question") or "").strip()
        prev_answer   = (data.get("prev_answer") or "").strip()
        prev_category = (data.get("prev_category") or "").strip()

        if not category or not question:
            return jsonify({"error": "Please provide category and question"}), 400

        mapping = {
            "1": "ikhtiar_hamil",
            "2": "sedang_hamil",
            "3": "lain_lain",
            "Ikhtiar Hamil": "ikhtiar_hamil",
            "Sedang Hamil": "sedang_hamil",
            "Lain-Lain": "lain_lain"
        }
        cat_key = mapping.get(str(category))
        if not cat_key:
            return jsonify({"error": "Unknown category"}), 400

        try:
            user_lang = detect(question)
        except Exception:
            user_lang = "ms"

        intent_why = _looks_like_why(question)
        intent_how = _looks_like_how(question)

        # contextual WHY: use previous turn if vague
        if intent_why:
            ctx_q, ctx_a, ctx_cat = None, None, None
            if prev_answer:
                ctx_q, ctx_a, ctx_cat = prev_question, prev_answer, (prev_category or cat_key)
            elif history:
                last_bot = next((m.get("content","") for m in reversed(history) if m.get("role") == "assistant"), "")
                last_user = next((m.get("content","") for m in reversed(history) if m.get("role") == "user"), "")
                ctx_q, ctx_a, ctx_cat = (last_user or question), last_bot, cat_key
            if ctx_a and len(_normalize(ctx_a)) > 8:
                extra = _justify_answer(ctx_q or question, ctx_a, ctx_cat)
                if extra:
                    suggestions = related_kb_questions(question, ctx_cat, n=3)
                    return jsonify({"answer": linkify_platforms(extra), "suggestions": suggestions, "used_context": True})

        # 1) KB retrieval in the selected category
        cands = retrieve_candidates(question, cat_key, top_n=TOP_N)
        best = cands[0] if cands else None
        selected_score = best["score"] if best else 0.0

        def _score_ok(r):
            if not r: return False
            cos = r.get("cos", 0.0)
            kw  = r.get("kw", 0.0)
            return (cos >= 0.26) or (cos >= COS_MIN and kw >= KW_MIN)

        selected_ok = _score_ok(best)

        # Always compute a global best for comparison
        global_best, global_cat = retrieve_candidates_any(question, top_n=TOP_N)
        global_score = global_best["score"] if global_best else 0.0

        # Decide whether to keep selected category or switch
        use_kb = False
        if best and selected_ok and (selected_score >= SELECTED_MIN) and (selected_score >= global_score - RESCUE_DELTA):
            use_kb = True
        else:
            if global_best and _score_ok(global_best) and (global_score - selected_score >= RESCUE_DELTA):
                best = global_best
                cat_key = global_cat
                use_kb = True
            else:
                use_kb = False

        if use_kb:
            need_selector = False
            if len(cands) >= 2 and cat_key == mapping.get(str(category)):  # only compare within same cat list we fetched
                margin = cands[0]["score"] - cands[1]["score"]
                need_selector = margin < 0.08

            if not need_selector:
                answer = best["a"].strip()
                weak_match = (best.get("kw", 0.0) < 0.12 and best.get("cos", 0.0) < 0.22)
                if (intent_why or intent_how) and (_looks_like_catalog(answer) or weak_match):

                    extra = _justify_answer(question, answer, cat_key)
                    if not extra:

                        any_best, any_cat = retrieve_candidates_any(question, top_n=TOP_N)
                        if any_cat:
                            kws = list(set(_tokens(question)))
                            facts = _gather_fact_context(any_cat, kws)
                            if facts:
                                expl_prompt = (
                                    "You are EffertyAskMe. The user asked 'why/how'. "
                                    "Use ONLY the KB facts below to give a brief rationale in the user's language. "
                                    "Avoid medical promises; 1–5 short sentences; end with: "
                                    "'Nasihat umum sahaja; dapatkan nasihat doktor jika ragu.'"
                                )
                                msgs = [
                                    {"role":"system","content": expl_prompt},
                                    {"role":"system","content": f"KB facts:\n{facts}"},
                                    {"role":"user","content": question}
                                ]
                                try:
                                    chat = client.chat.completions.create(
                                        model=CHAT_MODEL, messages=msgs, temperature=0.2, max_tokens=160
                                    )
                                    extra = _clean_prefixes((chat.choices[0].message.content or "").strip())
                                except Exception:
                                    extra = ""
                    if extra:
                        extra = linkify_platforms(extra)
                        suggestions = related_kb_questions(question, cat_key, exclude_q=best["q"], n=3)
                        return jsonify({"answer": extra, "suggestions": suggestions, "used_context": True})

                if JUSTIFY_MODE and _need_explanation(question, answer):
                    extra = _justify_answer(question, answer, cat_key)
                    if extra:
                        answer = f"{answer}\n\n{extra}"

                if user_lang == "en":
                    try:
                        answer = GoogleTranslator(source="auto", target="en").translate(answer)
                    except Exception as e:
                        print("Translation error:", e)

                answer = linkify_platforms(answer)
                suggestions = related_kb_questions(question, cat_key, exclude_q=best["q"], n=3)
                return jsonify({"answer": answer, "suggestions": suggestions, "used_context": False})

            # tie-breaker using model to pick ONE KB answer
            context_block = "\n\n".join([f"Question: {c['q']}\nAnswer: {c['a']}" for c in cands[:TOP_N]])
            system_prompt = (
                "You are EffertyAskMe. Choose exactly ONE QA pair from the KB that best answers the user.\n"
                "Return ONLY the FULL answer text for that pair, with identical wording and line breaks.\n"
                "Do not merge or summarize. Use the user's language.\n"
                "If nothing fits, reply exactly: 'Harap maaf, sila berhubung dengan agent kami untuk mengetahui lebih lanjut dengan menekan butang 4.'"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Knowledge Base:\n{context_block}"},
                {"role": "user", "content": f"User question ({user_lang}): {question}"}
            ]
            chat = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.0, max_tokens=1000)
            answer = _clean_prefixes(chat.choices[0].message.content.strip())

            if answer.strip().lower().startswith("harap maaf"):
                use_kb = False
            else:
                try:
                    if detect(question) == "en":
                        answer = GoogleTranslator(source="auto", target="en").translate(answer)
                except Exception:
                    pass
                answer = linkify_platforms(answer)
                suggestions = related_kb_questions(question, cat_key, n=3)
                return jsonify({"answer": answer, "suggestions": suggestions, "used_context": False})

        if STRICT_MODE:
            fallback = "Harap maaf. Sila berhubung dengan agent kami untuk mengetahui lebih lanjut dengan menekan butang 4."
            return jsonify({"answer": linkify_platforms(fallback), "suggestions": related_kb_questions(question, cat_key, n=3), "used_context": False})

        # Smart GPT fallback
        print("Smart GPT")
        kb_best, kb_cat = retrieve_candidates_any(question, TOP_N)
        kb_context = ""
        if kb_best:
            kb_context = f"Q: {kb_best['q']}\nA: {kb_best['a']}"

        sys = (
            "You are EffertyAskMe — an empathetic, knowledgeable assistant for Efferty Milk. "
            "Act like ChatGPT: reason clearly, be warm and practical, and answer in the SAME language as the user. "
            "Avoid medical claims/diagnosis. If price/promo/stock/uncertain → advise contacting our team."
        )
        msgs = [{"role": "system", "content": sys}]
        if kb_context:
            msgs.append({"role": "system", "content": f"KB Context (use only if relevant):\n{kb_context}"})
        for m in history[-6:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": question})

        chat = client.chat.completions.create(
            model=CHAT_MODEL, messages=msgs, temperature=0.7, max_tokens=600
        )
        answer = _clean_prefixes(chat.choices[0].message.content.strip())

        try:
            if detect(question) == "en":
                answer = GoogleTranslator(source="auto", target="en").translate(answer)
        except Exception:
            pass

        answer = f"{answer}\n\n{SMART_DISCLAIMER}"

        try:
            conn = _connect()
            c = conn.cursor()
            c.execute(
                "INSERT INTO generated_answers (category, user_question, ai_answer) VALUES (?, ?, ?)",
                (cat_key, question, answer)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

        suggestions = related_kb_questions(
            question, cat_key, exclude_q=best["q"] if best else None, n=3
        )
        return jsonify({
            "answer": linkify_platforms(answer),
            "suggestions": suggestions,
            "used_context": True
        })

    except Exception as e:
        print("Server error:", e)
        return jsonify({"error": "Internal Server Error"}), 500

# =========================
# Admin API
# =========================
@app.route("/admin/categories", methods=["GET"])
def list_categories():
    return jsonify({"categories": CATEGORIES})

@app.route("/admin/qa", methods=["GET"])
def admin_list_qa():
    try:
        category = request.args.get("category")
        query = (request.args.get("q") or "").strip()
        page = int(request.args.get("page", 1))
        limit = min(max(int(request.args.get("limit", 25)), 1), 200)
        offset = (page - 1) * limit

        conn = _connect()
        c = conn.cursor()

        base_sql = "FROM knowledge_base_qa WHERE 1=1"
        params = []
        if category:
            base_sql += " AND category=?"
            params.append(category)
        if query:
            base_sql += " AND (question LIKE ? OR answer LIKE ?)"
            like = f"%{query}%"
            params.extend([like, like])

        c.execute(f"SELECT COUNT(*) {base_sql}", params)
        total = c.fetchone()[0]

        c.execute(
            f"SELECT id, category, question, answer {base_sql} ORDER BY id DESC LIMIT ? OFFSET ?",
            (*params, limit, offset)
        )
        rows = c.fetchall()
        conn.close()

        items = [dict(id=r["id"], category=r["category"], question=r["question"], answer=r["answer"]) for r in rows]
        return jsonify({"items": items, "total": total, "page": page, "limit": limit})
    except Exception as e:
        print("Admin list error:", e)
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/admin/qa", methods=["POST"])
def admin_create_qa():
    try:
        data = request.get_json(force=True)
        category = data.get("category")
        question = (data.get("question") or "").strip()
        answer   = (data.get("answer") or "").strip()
        if not category or not question or not answer:
            return jsonify({"error": "Missing fields"}), 400
        if category not in CATEGORIES:
            return jsonify({"error": "Invalid category"}), 400

        emb = _embed(question).tobytes()
        conn = _connect()
        c = conn.cursor()
        c.execute(
            "INSERT INTO knowledge_base_qa (category, question, answer, q_embedding) VALUES (?, ?, ?, ?)",
            (category, question, answer, emb)
        )
        new_id = c.lastrowid
        conn.commit()
        conn.close()

        _sync_category_txt(category)
        return jsonify({"id": new_id, "category": category, "question": question, "answer": answer})
    except Exception as e:
        print("Admin create error:", e)
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/admin/qa/<int:item_id>", methods=["PUT"])
def admin_update_qa(item_id):
    try:
        data = request.get_json(force=True)
        category = data.get("category")
        question = data.get("question")
        answer   = data.get("answer")

        conn = _connect()
        c = conn.cursor()

        c.execute("SELECT category, question, answer FROM knowledge_base_qa WHERE id=?", (item_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return jsonify({"error": "Not found"}), 404

        old_cat = row["category"]
        new_cat = category if category else row["category"]
        new_q   = question.strip() if isinstance(question, str) else row["question"]
        new_a   = answer.strip() if isinstance(answer, str) else row["answer"]

        params = [new_cat, new_q, new_a]
        sql = "UPDATE knowledge_base_qa SET category=?, question=?, answer=?"
        if new_q != row["question"]:
            new_emb = _embed(new_q).tobytes()
            sql += ", q_embedding=?"
            params.append(new_emb)
        sql += " WHERE id=?"
        params.append(item_id)

        c.execute(sql, params)
        conn.commit()
        conn.close()

        if new_cat != old_cat:
            _sync_category_txt(old_cat)
        _sync_category_txt(new_cat)

        return jsonify({"id": item_id, "category": new_cat, "question": new_q, "answer": new_a})
    except Exception as e:
        print("Admin update error:", e)
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/admin/qa/<int:item_id>", methods=["DELETE"])
def admin_delete_qa(item_id):
    try:
        conn = _connect()
        c = conn.cursor()
        c.execute("SELECT category FROM knowledge_base_qa WHERE id=?", (item_id,))
        r = c.fetchone()
        if not r:
            conn.close()
            return jsonify({"error": "Not found"}), 404
        cat = r["category"]

        c.execute("DELETE FROM knowledge_base_qa WHERE id=?", (item_id,))
        conn.commit()
        conn.close()

        _sync_category_txt(cat)

        return jsonify({"ok": True})
    except Exception as e:
        print("Admin delete error:", e)
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)

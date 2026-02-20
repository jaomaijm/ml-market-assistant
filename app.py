import re
import json
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

# =========================================================
# Page setup
# =========================================================
st.set_page_config(page_title="Market Research Assistant", page_icon="🔎", layout="wide")
st.title("Market Research Assistant")
st.caption("Hi! I'm your market research assistant. Give me your keyword, I will find 5 relevant Wikipedia URLs and generate a report")

# =========================================================
# Session state defaults (MUST come before any usage)
# =========================================================
defaults = {
    "step": 1,  # 1 -> 2 -> 3
    "industry": "",
    "clarification": "",
    "final_query": "",
    "docs": [],
    "ranked": [],  # list of dicts: {title,url,score,doc}
    "confidence": "unknown",  # "high" | "low"
    "need_questions": "",
    "suggested_keywords": [],
    "keyword_pick": [],
    "keyword_typed": "",
    "forced_query": "",
    "show_keyword_picker": False,
    "selected_urls": [],
    "report": "",
    "api_key_session": "",
    "llm_option": "gpt-4.1-mini",
    "last_error": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# Sidebar: API key + (single-option) LLM dropdown + reset
# =========================================================
st.sidebar.header("API Key & LLM")

with st.sidebar.form("api_form", clear_on_submit=False):
    api_key = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
    llm_option = st.selectbox("Select LLM", options=["gpt-4.1-mini"], index=0)
    api_submitted = st.form_submit_button("Use this key", use_container_width=True)

st.sidebar.caption("Key is used only for this session and is not stored")
st.sidebar.markdown("[Get an API key](https://platform.openai.com/api-keys)")

if st.sidebar.button("Reset app", use_container_width=True):
    keep_key = st.session_state.get("api_key_session", "")
    keep_llm = st.session_state.get("llm_option", "gpt-4.1-mini")
    for k, v in defaults.items():
        st.session_state[k] = v
    st.session_state["api_key_session"] = keep_key
    st.session_state["llm_option"] = keep_llm
    st.rerun()

if api_submitted:
    st.session_state["api_key_session"] = api_key.strip()
    st.session_state["llm_option"] = llm_option

api_key_session = st.session_state.get("api_key_session", "")
if not api_key_session:
    st.info("Enter your OpenAI API key in the sidebar and click “Use this key” to begin")
    st.stop()

# =========================================================
# LLM init (guard)
# =========================================================
MODEL_NAME = "gpt-4.1-mini"
try:
    llm = ChatOpenAI(model=MODEL_NAME, openai_api_key=api_key_session, temperature=0.1)
except Exception as e:
    st.error("LLM init error. Please re-check your API key and try again")
    st.code(str(e))
    st.stop()

TOP_K = 5
LANG = "en"

STRONG_MATCH_MIN = 70
RELATED_MIN = 60

# =========================================================
# Safe utilities to prevent common Streamlit / parsing errors
# =========================================================
def safe_json_loads(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        return None

def safe_list_default(options, default_list):
    opt_set = set(options or [])
    cleaned = [x for x in (default_list or []) if x in opt_set]
    return cleaned

def safe_rerun_reset_error():
    st.session_state["last_error"] = ""
    st.rerun()

# =========================================================
# Helpers
# =========================================================
def stepper():
    s = int(st.session_state.get("step", 1))
    progress_value = {1: 1/3, 2: 2/3, 3: 1.0}.get(s, 1/3)
    st.progress(progress_value, text=f"Progress: Step {s} of 3")
    st.divider()

def looks_like_input(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 3:
        return False
    if re.fullmatch(r"[\W_]+", t):
        return False
    return True

# -------- NEW: detect gibberish / incomplete inputs (asked by you) --------
def is_probably_gibberish_or_incomplete(text: str):
    """
    Returns (bad: bool, reason: str)
    - gibberish like "ghdbfv"
    - incomplete/fragment like "electr"
    Heuristics only (cheap, fast). We also do a Wikipedia-retrieval sanity check below.
    """
    t = (text or "").strip()
    low = t.lower()

    # Too short to be meaningful
    if len(low) < 4:
        return True, "Too short. Please enter a full industry/topic (at least 4 characters)"

    # Mostly consonants (e.g., ghdbfv) => likely nonsense
    letters = re.findall(r"[a-z]", low)
    if len(letters) >= 5:
        vowels = sum(ch in "aeiou" for ch in letters)
        if vowels / max(1, len(letters)) < 0.25:
            return True, "Looks like random letters. Please type a real word or phrase (e.g., 'electric vehicles')"

    # Single long token that is not spaced and ends mid-word often
    # (common case: "electr", "cosmet", "automot")
    if " " not in low and len(low) <= 6:
        if low.endswith(("r", "t", "c", "m", "v", "n", "l")) and not low.endswith(("ing", "ion", "ics")):
            return True, "Looks like an incomplete word. Please type the full industry name (e.g., 'electricity' or 'electric vehicles')"

    return False, ""

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))

def extract_url(doc) -> str:
    md = getattr(doc, "metadata", {}) or {}
    return (md.get("source") or md.get("url") or "").strip()

def extract_title(doc) -> str:
    md = getattr(doc, "metadata", {}) or {}
    return (md.get("title") or md.get("page_title") or "Wikipedia page").strip()

def _normalize_query(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q

def _strip_parentheses(q: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*", " ", (q or "")).strip()

def retrieve_wikipedia_docs(query: str, top_k: int = 12):
    retriever = WikipediaRetriever(top_k_results=top_k, lang=LANG)
    return retriever.invoke(query)

# -------- NEW: sanity-check step 1 input via Wikipedia results --------
def wiki_sanity_check(industry_text: str) -> tuple[bool, str, list]:
    """
    Uses WikipediaRetriever to decide if the input has meaning.
    Returns (ok, message, docs_preview)
    """
    q = _normalize_query(industry_text)
    try:
        docs = retrieve_wikipedia_docs(q, top_k=5)
    except Exception:
        docs = []

    # If nothing retrieved, likely nonsense or too vague
    if not docs:
        return False, "I couldn't find any Wikipedia pages for that. Please re-type with a real industry/topic", []

    # If we got docs but titles look irrelevant (e.g., weird disambiguation / random), we treat as not ok
    titles = [extract_title(d).lower() for d in docs if d is not None]
    joined_titles = " | ".join(titles)

    # If too many titles are empty or generic
    if len(titles) < 2:
        return False, "I found too little to confirm what you mean. Please type a clearer industry/topic", docs

    # If query is a short fragment and doesn't appear in any title, treat as incomplete
    low = q.lower()
    if " " not in low and len(low) <= 6:
        if not any(low in t for t in titles):
            return False, "This looks like a partial word. Please type the full industry name (example: 'electric vehicles')", docs

    # Otherwise OK
    return True, "", docs

def retrieve_at_least_five_docs(primary_query: str, industry_fallback: str) -> list:
    primary_query = _normalize_query(primary_query)
    industry_fallback = _normalize_query(industry_fallback)

    candidates = []
    seen = set()

    def add_docs(docs):
        for d in docs or []:
            url = extract_url(d)
            title = extract_title(d)
            key = (url.lower(), title.lower())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(d)

    query_variants = []
    if primary_query:
        query_variants += [
            primary_query,
            _strip_parentheses(primary_query),
            f"{primary_query} industry",
            f"{primary_query} market",
        ]
    if industry_fallback:
        query_variants += [
            industry_fallback,
            f"{industry_fallback} industry",
            f"{industry_fallback} market",
        ]

    for q in query_variants:
        if len(candidates) >= TOP_K:
            break
        try:
            docs = retrieve_wikipedia_docs(q, top_k=12)
        except Exception:
            docs = []
        add_docs(docs)

    if len(candidates) < TOP_K:
        try:
            docs = retrieve_wikipedia_docs("Industry", top_k=12)
        except Exception:
            docs = []
        add_docs(docs)

    return candidates

def llm_rank_docs(industry_query: str, docs):
    items = []
    for d in docs or []:
        title = extract_title(d)
        url = extract_url(d)
        snippet = (getattr(d, "page_content", "") or "").strip().replace("\n", " ")
        snippet = snippet[:900]
        items.append({"title": title, "url": url, "snippet": snippet})

    if not items:
        return []

    judge_prompt = f"""
You are scoring relevance of Wikipedia pages to a user's market research topic.

User topic:
{industry_query}

Pages (JSON):
{json.dumps(items, ensure_ascii=False)}

Return ONLY valid JSON:
{{
  "scores": [
    {{"title":"...","url":"...","score": 0}},
    ...
  ]
}}

Rules:
- score is integer 0-100
- score 90-100: directly about the industry/market/topic
- score 60-89: related but broader / partial match
- score 0-59: weak match / mostly unrelated
- do NOT add extra keys
""".strip()

    scores = None
    try:
        raw = llm.invoke(judge_prompt).content.strip()
        data = safe_json_loads(raw)
        if data and isinstance(data.get("scores", None), list):
            scores = data["scores"]
    except Exception:
        scores = None

    if scores is None:
        scores = [{"title": it["title"], "url": it["url"], "score": 55} for it in items]

    score_map = {(s.get("title",""), s.get("url","")): int(s.get("score", 0)) for s in scores}

    ranked = []
    for d in docs or []:
        title = extract_title(d)
        url = extract_url(d)
        score = score_map.get((title, url), 50)
        ranked.append({"title": title, "url": url, "score": score, "doc": d})

    ranked.sort(key=lambda x: x["score"], reverse=True)

    if len(ranked) < TOP_K:
        for i in range(TOP_K - len(ranked)):
            ranked.append({
                "title": f"Placeholder (no strong Wikipedia match found) #{i+1}",
                "url": "",
                "score": 0,
                "doc": None
            })

    return ranked[:TOP_K]

def confidence_from_scores(ranked):
    if not ranked:
        return "low"
    scores = [r["score"] for r in ranked[:TOP_K] if isinstance(r.get("score", None), int)]
    if len(scores) < 3:
        return "low"
    avg = sum(scores) / len(scores)
    if avg >= 78 and min(scores) >= 65:
        return "high"
    return "low"

def build_query(industry: str, clarification: str) -> str:
    industry = (industry or "").strip()
    clarification = (clarification or "").strip()
    if clarification:
        return f"{industry} ({clarification})"
    return industry

def generate_clarifying_questions(industry: str, ranked):
    titles = [r["title"] for r in ranked[:TOP_K]]
    urls = [r["url"] for r in ranked[:TOP_K]]
    scores = [r["score"] for r in ranked[:TOP_K]]

    prompt = f"""
You are trying to avoid pulling the wrong Wikipedia pages for market research.

User input:
{industry}

Top pages found (title | score | url):
{chr(10).join([f"- {t} | {s} | {u}" for t,s,u in zip(titles, scores, urls)])}

Task:
Write 3–4 short clarifying questions to disambiguate the user's intent.
Also propose 6–10 possible clarification keywords/phrases (short chips) the user could click.

Return ONLY valid JSON:
{{
  "questions": ["...","..."],
  "keywords": ["...","..."]
}}
""".strip()

    qs, kws = None, None
    try:
        raw = llm.invoke(prompt).content.strip()
        data = safe_json_loads(raw)
        if data:
            qs = data.get("questions", [])
            kws = data.get("keywords", [])
    except Exception:
        qs, kws = None, None

    if not qs or not isinstance(qs, list):
        qs = [
            "Which country/region is this focused on",
            "Is this B2C, B2B, or both",
            "Which sub-segment do you mean (products, services, adoption, retail, etc.)",
        ]
    if not kws or not isinstance(kws, list):
        kws = ["UK", "Thailand", "B2C", "B2B", "Retail", "E-commerce", "Premium", "Mass market"]

    qs = [str(q).strip() for q in qs if str(q).strip()]
    kws = [str(k).strip() for k in kws if str(k).strip()]
    return qs[:4], kws[:10]

def enforce_under_500_words(text: str, limit: int = 500, max_rounds: int = 3) -> str:
    current = (text or "").strip()
    for _ in range(max_rounds):
        if word_count(current) <= limit:
            return current
        compress_prompt = f"""
Shorten this report to UNDER {limit} words.

Rules:
- Keep the SAME headings
- Keep bullets readable
- Remove low-importance details first
- Do NOT add new facts
- Output ONLY the revised text

Report:
{current}
""".strip()
        try:
            current = llm.invoke(compress_prompt).content.strip()
        except Exception:
            break
    tokens = current.split()
    return " ".join(tokens[:limit])

def write_report_from_docs(topic: str, docs):
    clean_docs = [d for d in (docs or []) if d is not None]
    context = "\n\n".join([(getattr(d, "page_content", "") or "") for d in clean_docs]).strip()

    if not context or len(context) < 200:
        return (
            "## Industry brief: " + (topic or "Unknown") + "\n\n"
            "### Scope\n"
            "Wikipedia did not return enough usable context for a reliable report based on your selected pages\n\n"
            "### Limits of this report\n"
            "Please adjust your keywords or pick different pages in Step 3\n"
        )

    prompt = f"""
You are a market research assistant writing a clean Markdown brief for a business analyst.

Topic:
{topic}

Hard rules:
- Use ONLY the Wikipedia context below
- Do NOT invent facts, numbers, competitors, trends, or claims not supported by the context
- STRICTLY under 500 words (target 420–480)
- Do NOT use separators like "---"
- Do NOT put headings on the same line as normal text
- Write with short paragraphs + bullets (very readable)

Output format (follow exactly):

## Industry brief: <clean industry name>

### Scope
1–2 sentences.

### Market offering
- 2–4 bullets

### Value chain
- 2–4 bullets

### Competitive landscape (categories only)
- 2–5 bullets

### Key trends and drivers
- 3–6 bullets

### Limits of this report
1–2 sentences.

Wikipedia context:
{context}
""".strip()

    try:
        out = llm.invoke(prompt).content.strip()
    except Exception:
        out = (
            "## Industry brief: " + (topic or "Unknown") + "\n\n"
            "### Scope\n"
            "I hit an LLM error while writing the report\n\n"
            "### Limits of this report\n"
            "Try regenerate or change selected pages\n"
        )

    return enforce_under_500_words(out)

def relevance_bucket(score: int) -> str:
    if score >= STRONG_MATCH_MIN:
        return "Strong match"
    if score >= RELATED_MIN:
        return "Related"
    return "Weak match"

# =========================================================
# Global error banner (non-blocking)
# =========================================================
if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])
    if st.button("Clear error message", use_container_width=True):
        safe_rerun_reset_error()

# =========================================================
# UI
# =========================================================
stepper()

# -----------------------------
# STEP 1
# -----------------------------
st.subheader("Step 1: Input")
industry_input = st.text_input(
    "Industry",
    value=st.session_state.get("industry", ""),
    placeholder="Example: pet companionship market, EV charging, cosmetics in Vietnam",
    label_visibility="collapsed",
)

col_a, col_b = st.columns(2)
with col_a:
    step1_go = st.button("Continue to Step 2", type="primary", use_container_width=True)
with col_b:
    clear_all = st.button("Clear results", use_container_width=True)

if clear_all:
    keep_key = st.session_state.get("api_key_session", "")
    keep_llm = st.session_state.get("llm_option", "gpt-4.1-mini")
    for k, v in defaults.items():
        st.session_state[k] = v
    st.session_state["api_key_session"] = keep_key
    st.session_state["llm_option"] = keep_llm
    st.rerun()

if step1_go:
    if not looks_like_input(industry_input):
        st.warning("Please enter an industry/topic first")
        st.stop()

    # NEW: step 1 validation for gibberish/incomplete BEFORE going to step 2
    bad, reason = is_probably_gibberish_or_incomplete(industry_input)
    if bad:
        st.warning(reason)
        st.info("Examples you can try: 'electric vehicles', 'pet companionship market', 'cosmetics in Vietnam'")
        st.stop()

    ok, msg, preview_docs = wiki_sanity_check(industry_input)
    if not ok:
        st.warning(msg)
        if preview_docs:
            st.markdown("I found these Wikipedia pages, but they don't clearly match what you typed. Try re-typing your topic")
            for d in preview_docs[:5]:
                st.markdown(f"- {extract_title(d)}")
        st.stop()

    st.session_state["industry"] = industry_input.strip()
    st.session_state["clarification"] = ""
    st.session_state["final_query"] = build_query(st.session_state["industry"], "")
    st.session_state["docs"] = []
    st.session_state["ranked"] = []
    st.session_state["confidence"] = "unknown"
    st.session_state["need_questions"] = ""
    st.session_state["suggested_keywords"] = []
    st.session_state["keyword_pick"] = []
    st.session_state["keyword_typed"] = ""
    st.session_state["forced_query"] = ""
    st.session_state["show_keyword_picker"] = False
    st.session_state["selected_urls"] = []
    st.session_state["report"] = ""
    st.session_state["last_error"] = ""
    st.session_state["step"] = 2
    st.rerun()

# -----------------------------
# STEP 2
# -----------------------------
st.subheader("Step 2: Find 5 relevant Wikipedia URLs")

if st.session_state.get("step", 1) >= 2 and st.session_state.get("industry", ""):
    st.markdown(f"**Current topic:** {st.session_state.get('final_query','').strip() or st.session_state.get('industry','').strip()}")
    st.caption(
        f"Relevance score guide: {STRONG_MATCH_MIN}–100 = Strong match, "
        f"{RELATED_MIN}–{STRONG_MATCH_MIN-1} = Related, 0–{RELATED_MIN-1} = Weak match"
    )

    if not st.session_state.get("ranked"):
        try:
            with st.spinner("Searching Wikipedia and ranking results..."):
                docs_pool = retrieve_at_least_five_docs(
                    primary_query=st.session_state["final_query"],
                    industry_fallback=st.session_state["industry"],
                )
                ranked = llm_rank_docs(st.session_state["final_query"], docs_pool)

            st.session_state["docs"] = docs_pool
            st.session_state["ranked"] = ranked
            st.session_state["confidence"] = confidence_from_scores(ranked)

            if st.session_state["confidence"] == "low":
                qs, kws = generate_clarifying_questions(st.session_state["final_query"], ranked)
                st.session_state["need_questions"] = "\n".join([f"- {q}" for q in qs])
                st.session_state["suggested_keywords"] = kws

        except Exception as e:
            st.session_state["last_error"] = "Step 2 failed while searching/ranking. Please retry"
            st.code(str(e))
            st.stop()

    ranked = st.session_state.get("ranked", [])[:TOP_K]

    if st.session_state.get("confidence") == "high":
        st.success("I’m confident these pages match your topic. Here are the top results with relevance scores")

        for i, r in enumerate(ranked, 1):
            st.markdown(f"**{i}. {r['title']}**  \n{r['url'] or '(no url available)'}")
            score = int(r.get("score", 0))
            st.progress(min(max(score, 0), 100) / 100.0, text=f"Relevance: {score}/100 ({relevance_bucket(score)})")

        go_step3 = st.button("Continue to Step 3", type="primary", use_container_width=True)
        if go_step3:
            st.session_state["selected_urls"] = [r["url"] for r in ranked if r.get("url")]
            st.session_state["step"] = 3
            st.rerun()

    if st.session_state.get("confidence") == "low":
        st.warning("I’m not confident the pages fully match what you mean. Help me narrow it down")

        if st.session_state.get("need_questions"):
            st.markdown("**Quick questions (so I don’t pull the wrong pages):**")
            st.markdown(st.session_state["need_questions"])

        st.markdown("**Optional: click a suggestion or type your own clarification**")
        suggestions = st.session_state.get("suggested_keywords", [])
        default_pick = safe_list_default(suggestions, st.session_state.get("keyword_pick", []))

        pick = st.multiselect(
            "Click keywords (optional)",
            options=suggestions,
            default=default_pick,
            key="clarify_pick",
        )
        typed = st.text_input(
            "Or type clarification (optional)",
            value=st.session_state.get("clarification", ""),
            placeholder="Example: UK, B2C, pet adoption, companion animals, dog walking, therapy animals",
            key="clarify_typed",
        )

        col1, col2 = st.columns(2)
        with col1:
            retry_with_context = st.button("Retry search with clarification", type="primary", use_container_width=True)
        with col2:
            force_keywords = st.button("Still unsure → show forced keyword options", use_container_width=True)

        if retry_with_context:
            st.session_state["keyword_pick"] = pick
            st.session_state["clarification"] = (typed or "").strip()

            combo = " ".join([st.session_state["clarification"]] + st.session_state["keyword_pick"]).strip()
            st.session_state["final_query"] = build_query(st.session_state["industry"], combo)

            try:
                with st.spinner("Re-searching Wikipedia and re-ranking..."):
                    docs_pool2 = retrieve_at_least_five_docs(
                        primary_query=st.session_state["final_query"],
                        industry_fallback=st.session_state["industry"],
                    )
                    ranked2 = llm_rank_docs(st.session_state["final_query"], docs_pool2)

                st.session_state["docs"] = docs_pool2
                st.session_state["ranked"] = ranked2
                st.session_state["confidence"] = confidence_from_scores(ranked2)

                if st.session_state["confidence"] == "low":
                    st.session_state["show_keyword_picker"] = True
                    qs, kws = generate_clarifying_questions(st.session_state["final_query"], ranked2)
                    st.session_state["need_questions"] = "\n".join([f"- {q}" for q in qs])
                    st.session_state["suggested_keywords"] = kws

                st.rerun()

            except Exception as e:
                st.session_state["last_error"] = "Retry search failed. Please try different clarification"
                st.code(str(e))
                st.stop()

        if force_keywords:
            st.session_state["show_keyword_picker"] = True
            st.rerun()

        if st.session_state.get("show_keyword_picker"):
            st.divider()
            st.subheader("Forced keywords (Round 2)")
            st.caption("Click suggestions, type your own, or do both")

            suggestions2 = st.session_state.get("suggested_keywords", [])
            default_pick2 = safe_list_default(suggestions2, st.session_state.get("keyword_pick", []))

            pick2 = st.multiselect(
                "Click keywords (optional)",
                options=suggestions2,
                default=default_pick2,
                key="force_pick",
            )
            typed2 = st.text_input(
                "Or type a forced keyword (optional)",
                value=st.session_state.get("keyword_typed", ""),
                placeholder="Example: Companion animal, Pet adoption, Dog walking, Therapy dog",
                key="force_typed",
            )

            force_go = st.button("Run forced search", type="primary", use_container_width=True)

            if force_go:
                st.session_state["keyword_pick"] = pick2
                st.session_state["keyword_typed"] = (typed2 or "").strip()

                forced_bits = []
                if st.session_state["keyword_pick"]:
                    forced_bits += st.session_state["keyword_pick"]
                if st.session_state["keyword_typed"]:
                    forced_bits.append(st.session_state["keyword_typed"])
                forced_phrase = " ".join(forced_bits).strip()

                st.session_state["forced_query"] = (
                    build_query(st.session_state["industry"], forced_phrase)
                    if forced_phrase
                    else st.session_state["final_query"]
                )

                try:
                    with st.spinner("Searching Wikipedia using forced keywords and ranking results..."):
                        docs_pool3 = retrieve_at_least_five_docs(
                            primary_query=st.session_state["forced_query"],
                            industry_fallback=st.session_state["industry"],
                        )
                        ranked3 = llm_rank_docs(st.session_state["forced_query"], docs_pool3)

                    st.session_state["docs"] = docs_pool3
                    st.session_state["ranked"] = ranked3
                    st.session_state["confidence"] = "low"
                    st.rerun()

                except Exception as e:
                    st.session_state["last_error"] = "Forced search failed. Try different forced keyword"
                    st.code(str(e))
                    st.stop()

            if st.session_state.get("forced_query") and st.session_state.get("ranked"):
                st.divider()
                st.markdown(f"**Forced topic used:** {st.session_state.get('forced_query','').strip()}")

                ranked_forced = st.session_state.get("ranked", [])[:TOP_K]
                url_options = []
                for i, r in enumerate(ranked_forced, 1):
                    title, url, score = r.get("title","Wikipedia page"), r.get("url",""), int(r.get("score", 0))
                    label = f"{i}. {title} ({score}/100)"
                    url_options.append((label, url))
                    st.markdown(f"**{label}**  \n{url or '(no url available)'}")
                    st.progress(min(max(score, 0), 100) / 100.0, text=f"Relevance: {score}/100 ({relevance_bucket(score)})")

                labels = [lbl for (lbl, _) in url_options]
                default_labels = labels[:]
                chosen_labels = st.multiselect(
                    "Select pages to carry into Step 3 (recommended: pick 3–5)",
                    options=labels,
                    default=safe_list_default(labels, default_labels),
                    key="forced_select_pages",
                )
                chosen_urls = [u for (lbl, u) in url_options if (lbl in chosen_labels and u)]
                st.session_state["selected_urls"] = chosen_urls

                go_step3b = st.button("Continue to Step 3", type="primary", use_container_width=True)
                if go_step3b:
                    st.session_state["step"] = 3
                    st.rerun()

# -----------------------------
# STEP 3
# -----------------------------
st.subheader("Step 3: Report (under 500 words)")

if st.session_state.get("step", 1) >= 3:
    topic_for_report = (
        st.session_state.get("forced_query", "").strip()
        or st.session_state.get("final_query", "").strip()
        or st.session_state.get("industry", "").strip()
    )

    ranked_all = st.session_state.get("ranked", [])[:TOP_K]
    if not ranked_all:
        st.info("Complete Step 2 first to retrieve Wikipedia pages")
        st.stop()

    st.caption(
        f"How to choose pages: {STRONG_MATCH_MIN}–100 = Strong match, "
        f"{RELATED_MIN}–{STRONG_MATCH_MIN-1} = Related, 0–{RELATED_MIN-1} = Weak match. "
        "If scores are low, choose pages that best match your exact meaning"
    )
    st.markdown(f"**Report topic:** {topic_for_report}")

    url_options = []
    for i, r in enumerate(ranked_all, 1):
        title = r.get("title", "Wikipedia page")
        url = r.get("url", "")
        score = int(r.get("score", 0))
        label = f"{i}. {title} ({score}/100 • {relevance_bucket(score)})"
        url_options.append((label, url))

    option_labels = [lbl for (lbl, _) in url_options]
    carried = st.session_state.get("selected_urls", [])
    default_labels = [lbl for (lbl, u) in url_options if (u and u in carried)] if carried else [lbl for (lbl, u) in url_options if u]

    chosen_labels_step3 = st.multiselect(
        "Select pages (3–5 recommended)",
        options=option_labels,
        default=safe_list_default(option_labels, default_labels),
        key="step3_picker",
    )

    chosen_urls_step3 = [u for (lbl, u) in url_options if (lbl in chosen_labels_step3 and u)]
    st.session_state["selected_urls"] = chosen_urls_step3

    if not chosen_urls_step3:
        st.warning("Please select at least 1 page (with a valid URL) to generate the report")
        st.stop()

    url_set = set(chosen_urls_step3)
    docs_to_use = []
    for r in ranked_all:
        if r.get("url") in url_set and r.get("doc") is not None:
            docs_to_use.append(r["doc"])

    gen_col1, gen_col2 = st.columns(2)
    with gen_col1:
        gen_report = st.button(
            "Generate report",
            type="primary",
            use_container_width=True,
            disabled=bool(st.session_state.get("report")),
        )
    with gen_col2:
        regen_report = st.button(
            "Regenerate",
            use_container_width=True,
            disabled=not bool(st.session_state.get("report")),
        )

    if regen_report:
        st.session_state["report"] = ""
        st.rerun()

    if gen_report:
        try:
            with st.spinner("Writing report..."):
                report = write_report_from_docs(topic_for_report, docs_to_use)
            st.session_state["report"] = report
            st.rerun()
        except Exception as e:
            st.session_state["last_error"] = "Step 3 report generation failed. Try regenerate or change selected pages"
            st.code(str(e))
            st.stop()

    if st.session_state.get("report"):
        st.markdown("### Report")
        st.markdown(st.session_state["report"])
        wc = word_count(st.session_state["report"])
        st.caption(f"Word count: {wc} / 500")

        st.download_button(
            "Download report (TXT)",
            data=st.session_state["report"],
            file_name="industry_report.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_report_txt",
        )

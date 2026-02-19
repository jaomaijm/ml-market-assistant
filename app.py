import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

# ==========================================================
# PAGE SETUP
# ==========================================================
st.set_page_config(page_title="Market Research Assistant", page_icon="🔎", layout="wide")
st.title("Market Research Assistant")
st.caption(
    "Step 1: input industry → Step 2: find top Wikipedia pages with relevance scores → Step 3: generate a ≤500-word report using ONLY those pages"
)

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("Settings")

api_key = st.sidebar.text_input("OpenAI API key", type="password", placeholder="sk-...")
model_choice = st.sidebar.selectbox("LLM", ["gpt-4.1-mini"], index=0)

st.sidebar.caption("Key is used only in this session and is not stored")

if st.sidebar.button("Reset app", use_container_width=True):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to begin")
    st.stop()

llm = ChatOpenAI(model=model_choice, temperature=0.2, openai_api_key=api_key)

# ==========================================================
# CONSTANTS
# ==========================================================
LANG = "en"
RETRIEVE_TOP_K = 10  # pull more, then rank to pick best 5
STRONG_THRESHOLD = 70  # confidence threshold for a page score
CONFIDENT_STRONG_MIN = 5  # if >= 5 strong pages, we treat it as confident

# ==========================================================
# HELPERS
# ==========================================================
def show_banner():
    if st.session_state.banner_text:
        kind = st.session_state.banner_kind or "info"
        msg = st.session_state.banner_text
        if kind == "success":
            st.success(msg)
        elif kind == "warning":
            st.warning(msg)
        elif kind == "error":
            st.error(msg)
        else:
            st.info(msg)
        st.session_state.banner_text = ""
        st.session_state.banner_kind = ""

def looks_like_input(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 3:
        return False
    if re.fullmatch(r"[\W_]+", t):
        return False
    return True

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))

def extract_title(doc) -> str:
    md = getattr(doc, "metadata", {}) or {}
    return md.get("title") or md.get("page_title") or "Wikipedia page"

def extract_url(doc) -> str:
    md = getattr(doc, "metadata", {}) or {}
    return md.get("source") or md.get("url") or ""

def retrieve_docs(query: str):
    retriever = WikipediaRetriever(top_k_results=RETRIEVE_TOP_K, lang=LANG)
    return retriever.invoke(query)

def safe_int(s: str, default: int = 0) -> int:
    nums = re.findall(r"\d+", s or "")
    if not nums:
        return default
    try:
        return int(nums[0])
    except:
        return default

def score_doc(query: str, doc) -> int:
    title = extract_title(doc)
    snippet = (getattr(doc, "page_content", "") or "")[:1400]
    prompt = f"""
Score relevance from 0 to 100.

User query:
{query}

Wikipedia page title:
{title}

Wikipedia content snippet:
{snippet}

Return ONLY one integer 0-100.
"""
    resp = llm.invoke(prompt).content.strip()
    score = safe_int(resp, 0)
    return max(0, min(100, score))

def rank_docs(query: str, docs):
    ranked = []
    for d in docs:
        ranked.append(
            {
                "score": score_doc(query, d),
                "title": extract_title(d),
                "url": extract_url(d),
                "doc": d,
            }
        )
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked

def build_query(industry: str, clarification: str = "", forced_kw: str = "") -> str:
    parts = [industry.strip()]
    if clarification.strip():
        parts.append(clarification.strip())
    if forced_kw.strip():
        parts.append(forced_kw.strip())
    return " ".join(parts).strip()

def strong_count_top5(ranked):
    top5 = ranked[:5]
    return len([x for x in top5 if x["score"] >= STRONG_THRESHOLD])

def suggest_forcing_keywords(industry: str, clarification: str) -> list:
    prompt = f"""
Generate 8 forcing keywords to disambiguate Wikipedia retrieval.

Base term:
{industry}

Clarification (if any):
{clarification}

Rules:
- 2 to 5 words each
- very specific interpretations / subsegments
- Wikipedia-search-friendly
Return ONLY as a comma-separated list.
"""
    resp = llm.invoke(prompt).content
    kws = [x.strip() for x in resp.split(",") if x.strip()]
    # de-dup while keeping order
    seen = set()
    out = []
    for kw in kws:
        k = kw.lower()
        if k not in seen:
            seen.add(k)
            out.append(kw)
    return out[:8]

def render_ranked_urls(ranked):
    if not ranked:
        st.warning("No pages found")
        return

    top5 = ranked[:5]
    strong = [x for x in top5 if x["score"] >= STRONG_THRESHOLD]
    backup = [x for x in top5 if x["score"] < STRONG_THRESHOLD]

    if len(strong) >= 5:
        st.success("Path 1: Confident — 5 strong matches found")
    else:
        st.warning(
            f"Path 2: Not fully confident — strong matches: {len(strong)}/5. "
            "Showing best available pages + backups with relevance scores"
        )

    st.caption("Relevance score is 0–100 (higher = more relevant). Strong match = 70+")
    st.markdown(f"Query used: `{st.session_state.query_used}`")

    if strong:
        st.markdown("### Strong matches")
        for i, x in enumerate(strong, 1):
            st.markdown(f"**{i}. {x['title']}**")
            st.caption(f"Relevance: {x['score']}/100")
            st.markdown(x["url"] or "_No URL available_")
            st.markdown("")

    if backup:
        st.markdown("### Backups (less direct)")
        for i, x in enumerate(backup, 1):
            st.markdown(f"**{i}. {x['title']}**")
            st.caption(f"Relevance: {x['score']}/100")
            st.markdown(x["url"] or "_No URL available_")
            st.markdown("")

def generate_report(industry: str, clarification: str, forced_kw: str, docs_for_report):
    # Topic should reflect latest intent
    topic_bits = [industry.strip()]
    if clarification.strip():
        topic_bits.append(f"context: {clarification.strip()}")
    if forced_kw.strip():
        topic_bits.append(f"focus: {forced_kw.strip()}")
    topic = " | ".join(topic_bits).strip()

    context = "\n\n".join([(d.page_content or "")[:2000] for d in docs_for_report]).strip()

    prompt = f"""
You are a market research assistant writing a clean, readable Markdown brief.

Topic (follow this intent exactly):
{topic}

Hard rules:
- Use ONLY the Wikipedia context below
- Do NOT invent facts, numbers, competitors, or claims not supported by the context
- STRICTLY under 500 words (target 420–480)
- Use headings + short paragraphs + bullets
- Headings on their own line
- No separators like "---"

Output format (follow exactly):

## Industry Brief: <clean industry name>

### Scope
Write 1–2 sentences as a normal paragraph.

### Market Offering
- Bullet
- Bullet

### Value Chain
- Bullet
- Bullet

### Competitive Landscape
- Bullet
- Bullet

### Key Trends and Drivers
- Bullet
- Bullet

### Limits
Write 1–2 sentences as a normal paragraph.

Wikipedia context:
{context}
"""
    out = llm.invoke(prompt).content.strip()
    if word_count(out) > 500:
        out = " ".join(out.split()[:500])
    return out

# ==========================================================
# TOP UX
# ==========================================================
def stepper():
    cols = st.columns(3)
    s = st.session_state.step
    cols[0].markdown(f"**{'➡️' if s == 1 else '✅' if s > 1 else '⬜'} Step 1: Input**")
    cols[1].markdown(f"**{'➡️' if s == 2 else '✅' if s > 2 else '⬜'} Step 2: URLs**")
    cols[2].markdown(f"**{'➡️' if s == 3 else '⬜'} Step 3: Report**")
    st.divider()

stepper()
show_banner()

# ==========================================================
# STEP 1: USER INPUT
# ==========================================================
st.subheader("Step 1: User input")
industry_input = st.text_input(
    "Industry",
    value=st.session_state.industry,
    placeholder="Example: pet market, semiconductors, cosmetics in Vietnam",
)

go_step2 = st.button("Go to Step 2: Find URLs", type="primary", use_container_width=True)

if go_step2:
    if not looks_like_input(industry_input):
        st.warning("Please enter a clearer industry term")
        st.stop()

    st.session_state.industry = industry_input.strip()
    st.session_state.step = 2

    # reset downstream
    st.session_state.clarification = ""
    st.session_state.forced_kw = ""
    st.session_state.forced_kw_text = ""
    st.session_state.query_used = ""
    st.session_state.ranked = []
    st.session_state.confidence_ok = False
    st.session_state.needs_clarification = False
    st.session_state.needs_forcing = False
    st.session_state.kw_suggestions = []
    st.session_state.report = ""
    st.session_state.last_action = ""

    st.rerun()

st.divider()

# ==========================================================
# STEP 2: FIND URLS
# ==========================================================
st.subheader("Step 2: Find 5 relevant URLs for writing report")

if st.session_state.step < 2:
    st.info("Complete Step 1 first")
else:
    # Base search button (first attempt)
    base_col1, base_col2 = st.columns([1, 1])
    with base_col1:
        run_base = st.button("Find URLs (first attempt)", type="primary", use_container_width=True)
    with base_col2:
        rerun_step2 = st.button("Re-run Step 2 from scratch", use_container_width=True)

    if rerun_step2:
        st.session_state.clarification = ""
        st.session_state.forced_kw = ""
        st.session_state.forced_kw_text = ""
        st.session_state.query_used = ""
        st.session_state.ranked = []
        st.session_state.confidence_ok = False
        st.session_state.needs_clarification = False
        st.session_state.needs_forcing = False
        st.session_state.kw_suggestions = []
        st.session_state.report = ""
        st.session_state.last_action = ""
        st.rerun()

    if run_base:
        q = build_query(st.session_state.industry)
        st.session_state.query_used = q

        with st.spinner("Retrieving Wikipedia pages..."):
            docs = retrieve_docs(q)

        with st.spinner("Ranking relevance (0–100)..."):
            ranked = rank_docs(q, docs)

        st.session_state.ranked = ranked
        st.session_state.last_action = "base"

        strong_top5 = strong_count_top5(ranked)
        if strong_top5 >= CONFIDENT_STRONG_MIN:
            st.session_state.confidence_ok = True
            st.session_state.needs_clarification = False
            st.session_state.needs_forcing = False
            st.session_state.banner_kind = "success"
            st.session_state.banner_text = "Confident match found. Review URLs below then go to Step 3"
        else:
            st.session_state.confidence_ok = False
            st.session_state.needs_clarification = True
            st.session_state.needs_forcing = False
            st.session_state.banner_kind = "warning"
            st.session_state.banner_text = "Not confident. Please add clarification to avoid pulling wrong URLs"
        st.rerun()

    # Always show results if we have them
    if st.session_state.ranked:
        render_ranked_urls(st.session_state.ranked)

    # Path2: Ask for clarification (only if not confident)
    if st.session_state.needs_clarification:
        st.markdown("### Path 2.1: Add clarification (to avoid wrong URLs)")
        st.caption("Add country / sub-segment / interpretation / B2B-B2C, etc.")

        st.session_state.clarification = st.text_input(
            "Clarification",
            value=st.session_state.clarification,
            placeholder="Example: UK, pet companionship, B2C retail, premium segment",
            key="clarification_box",
        )

        run_clarified = st.button("Search again with clarification", type="primary", use_container_width=True)

        if run_clarified:
            if not st.session_state.clarification.strip():
                st.warning("Please add at least 1 short clarification")
                st.stop()

            q = build_query(st.session_state.industry, clarification=st.session_state.clarification)
            st.session_state.query_used = q

            with st.spinner("Retrieving Wikipedia pages..."):
                docs = retrieve_docs(q)

            with st.spinner("Ranking relevance (0–100)..."):
                ranked = rank_docs(q, docs)

            st.session_state.ranked = ranked
            st.session_state.last_action = "clarified"

            strong_top5 = strong_count_top5(ranked)
            if strong_top5 >= CONFIDENT_STRONG_MIN:
                st.session_state.confidence_ok = True
                st.session_state.needs_clarification = False
                st.session_state.needs_forcing = False
                st.session_state.banner_kind = "success"
                st.session_state.banner_text = "Now confident. Review URLs below then go to Step 3"
            else:
                st.session_state.confidence_ok = False
                st.session_state.needs_clarification = False
                st.session_state.needs_forcing = True
                st.session_state.kw_suggestions = suggest_forcing_keywords(
                    st.session_state.industry, st.session_state.clarification
                )
                st.session_state.banner_kind = "warning"
                st.session_state.banner_text = "Still unsure. Use forcing keywords below (click or type) to guide the system"
            st.rerun()

    # Path2.3: forcing keywords (click OR type), then show URLs + scores
    if st.session_state.needs_forcing:
        st.markdown("### Path 2.3: Forcing keywords (click or type)")
        st.caption(
            "Pick a specific keyword to force retrieval. After forcing, the system will show the best 5 pages it can find. "
            "If only 3 strong pages exist, it will show those 3 + 2 weaker backups with scores"
        )

        # clickable chips
        if st.session_state.kw_suggestions:
            cols = st.columns(4)
            for i, kw in enumerate(st.session_state.kw_suggestions):
                if cols[i % 4].button(kw, use_container_width=True, key=f"kw_chip_{i}"):
                    st.session_state.forced_kw = kw
                    st.session_state.forced_kw_text = ""
                    q = build_query(
                        st.session_state.industry,
                        clarification=st.session_state.clarification,
                        forced_kw=st.session_state.forced_kw,
                    )
                    st.session_state.query_used = q

                    with st.spinner("Retrieving Wikipedia pages..."):
                        docs = retrieve_docs(q)

                    with st.spinner("Ranking relevance (0–100)..."):
                        ranked = rank_docs(q, docs)

                    st.session_state.ranked = ranked
                    st.session_state.last_action = "forced"
                    st.session_state.banner_kind = "info"
                    st.session_state.banner_text = "Forced search completed. Review URLs and scores below, then go to Step 3"
                    st.rerun()

        # typed forcing keyword
        st.session_state.forced_kw_text = st.text_input(
            "Or type your own forcing keyword",
            value=st.session_state.forced_kw_text,
            placeholder="Example: pet companionship, pet industry, companion animal",
            key="kw_type_box",
        )
        run_forced_typed = st.button("Force search (typed keyword)", type="primary", use_container_width=True)

        if run_forced_typed:
            kw = st.session_state.forced_kw_text.strip()
            if not kw:
                st.warning("Type a forcing keyword or click one of the chips")
                st.stop()

            st.session_state.forced_kw = kw
            q = build_query(
                st.session_state.industry,
                clarification=st.session_state.clarification,
                forced_kw=st.session_state.forced_kw,
            )
            st.session_state.query_used = q

            with st.spinner("Retrieving Wikipedia pages..."):
                docs = retrieve_docs(q)

            with st.spinner("Ranking relevance (0–100)..."):
                ranked = rank_docs(q, docs)

            st.session_state.ranked = ranked
            st.session_state.last_action = "forced"
            st.session_state.banner_kind = "info"
            st.session_state.banner_text = "Forced search completed. Review URLs and scores below, then go to Step 3"
            st.rerun()

    st.divider()

    # Move to Step 3 when we have at least some ranked pages
    to_step3 = st.button("Go to Step 3: Generate report", type="primary", use_container_width=True)
    if to_step3:
        if not st.session_state.ranked:
            st.warning("Please run Step 2 first to retrieve URLs")
            st.stop()
        st.session_state.step = 3
        st.rerun()

# ==========================================================
# STEP 3: REPORT
# ==========================================================
st.subheader("Step 3: Report (≤ 500 words)")

if st.session_state.step < 3:
    st.info("Complete Step 2 first")
else:
    ranked = st.session_state.ranked
    top5 = ranked[:5]
    docs_for_report = [x["doc"] for x in top5 if x.get("doc")]

    strong_in_top5 = [x for x in top5 if x["score"] >= STRONG_THRESHOLD]
    backup_in_top5 = [x for x in top5 if x["score"] < STRONG_THRESHOLD]

    if not top5:
        st.error("No pages available to write a report")
        st.stop()

    # transparent statement like you asked
    if len(strong_in_top5) >= 5:
        st.success("Using 5 strong pages to generate the report")
    else:
        st.warning(
            f"Using best available pages: strong = {len(strong_in_top5)} page(s), backups = {len(backup_in_top5)} page(s). "
            "This is the best Wikipedia match the system can find for your query"
        )

    st.markdown("### Pages used")
    for i, x in enumerate(top5, 1):
        st.markdown(f"**{i}. {x['title']}**")
        st.caption(f"Relevance: {x['score']}/100")
        st.markdown(x["url"] or "_No URL available_")
        st.markdown("")

    gen1, gen2 = st.columns([1, 1])
    with gen1:
        gen_report = st.button("Generate report", type="primary", use_container_width=True)
    with gen2:
        regen_report = st.button("Regenerate", use_container_width=True)

    if regen_report:
        st.session_state.report = ""
        st.rerun()

    if gen_report:
        with st.spinner("Generating report..."):
            st.session_state.report = generate_report(
                st.session_state.industry,
                st.session_state.clarification,
                st.session_state.forced_kw,
                docs_for_report,
            )
        st.session_state.banner_kind = "success"
        st.session_state.banner_text = "Report generated"
        st.rerun()

    if st.session_state.report:
        st.markdown("### Report")
        st.markdown(st.session_state.report)
        wc = word_count(st.session_state.report)
        st.caption(f"Word count: {wc} / 500")

        st.download_button(
            "Download report (TXT)",
            data=st.session_state.report,
            file_name="industry_report.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_report_txt",
        )

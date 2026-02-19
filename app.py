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
    "Enter an industry. I will retrieve Wikipedia pages and produce a report under 500 words based only on those pages. "
    "If I cannot confidently identify the right pages, I will ask for more details instead of guessing."
)

# ==========================================================
# SESSION STATE DEFAULTS
# ==========================================================
defaults = {
    "stage": "step1",
    "industry": "",
    "clarification_1": "",
    "clarification_2": "",
    "docs": [],
    "ranked_docs": [],

    "confidence_mode": "unknown",
    "clarify_round": 0,

    "context_ideas": [],
    "force_keywords": [],
    "force_keyword_selected": "",
    "force_keyword_text": "",

    "query_used": "",
    "topic_for_report": "",        # NEW: always reflect the latest refined user intent
    "report": "",

    "banner_text": "",
    "banner_kind": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("API Key & LLM")
api_key = st.sidebar.text_input("OpenAI API key", type="password", placeholder="sk-...")

# Only one model in dropdown
model_choice = st.sidebar.selectbox("Select LLM", ["gpt-4.1-mini"], index=0)

st.sidebar.caption("Your key is used only in this session and is not stored")

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
RETRIEVE_TOP_K = 10
STRONG_THRESHOLD = 70
MIN_STRONG_FOR_AUTO = 5

# ==========================================================
# HELPERS
# ==========================================================
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

def safe_int_from_text(s: str, default: int = 0) -> int:
    nums = re.findall(r"\d+", s or "")
    if not nums:
        return default
    try:
        return int(nums[0])
    except:
        return default

def score_relevance(query: str, doc) -> int:
    title = extract_title(doc)
    snippet = (doc.page_content or "")[:1600]
    judge_prompt = f"""
Score relevance from 0 to 100.

User industry query:
{query}

Wikipedia page title:
{title}

Wikipedia content snippet:
{snippet}

Rules:
- Return ONLY one integer 0-100
"""
    resp = llm.invoke(judge_prompt).content.strip()
    score = safe_int_from_text(resp, default=0)
    return max(0, min(100, score))

def rank_docs(query: str, docs):
    ranked = []
    for d in docs:
        s = score_relevance(query, d)
        ranked.append({
            "score": s,
            "doc": d,
            "title": extract_title(d),
            "url": extract_url(d),
        })
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked

def is_confident(ranked):
    strong = [x for x in ranked[:10] if x["score"] >= STRONG_THRESHOLD]
    return len(strong) >= MIN_STRONG_FOR_AUTO

def suggest_context_ideas(industry: str):
    prompt = f"""
Generate 6 short clickable context ideas that would clarify this market research request.
Keep each idea under 6 words.

Industry term:
{industry}

Return ONLY as a comma-separated list.
"""
    resp = llm.invoke(prompt).content
    ideas = [x.strip() for x in resp.split(",") if x.strip()]
    return ideas[:6] if ideas else []

def suggest_force_keywords(industry: str, clar1: str, clar2: str):
    prompt = f"""
Generate 8 specific sub-segment keywords for Wikipedia search to disambiguate the request.

Base industry:
{industry}

Clarification 1:
{clar1}

Clarification 2:
{clar2}

Rules:
- Keywords must be short (2-5 words)
- Wikipedia-search friendly
- Avoid duplicates
Return ONLY as comma-separated list.
"""
    resp = llm.invoke(prompt).content
    kws = [x.strip() for x in resp.split(",") if x.strip()]
    return kws[:8] if kws else []

def build_query(industry: str, c1: str = "", c2: str = "", forced_kw: str = "") -> str:
    parts = [industry.strip()]
    if c1.strip():
        parts.append(c1.strip())
    if c2.strip():
        parts.append(c2.strip())
    if forced_kw.strip():
        parts.append(forced_kw.strip())
    return " ".join(parts).strip()

def compute_topic_for_report(industry: str, c1: str, c2: str, forced_kw: str) -> str:
    """
    NEW: Make the report topic reflect the latest refined user intent.
    Example:
    industry="pet", forced_kw="pet companionship" -> "pet — pet companionship"
    """
    bits = [industry.strip()]
    # prefer "forced" as the strongest signal of user intent if present
    if forced_kw.strip():
        bits.append(f"focus: {forced_kw.strip()}")
    if c1.strip():
        bits.append(f"context: {c1.strip()}")
    if c2.strip():
        bits.append(f"more context: {c2.strip()}")
    return " | ".join(bits).strip()

def generate_report(topic_for_report: str, docs_for_report):
    context = "\n\n".join([(d.page_content or "")[:2000] for d in docs_for_report]).strip()

    prompt = f"""
You are a professional market research analyst writing a clean, readable Markdown brief.

Topic (follow this intent exactly):
{topic_for_report}

Hard rules:
- Use ONLY the context below
- Do NOT invent facts, numbers, competitors, or claims not supported by the context
- STRICTLY under 500 words (target 420–480)
- Do NOT put headings and normal text on the same line
- Use headings + short paragraphs + bullets
- No separators like "---"

Output format (follow exactly):

## Industry Brief: <clean industry name based on topic above>

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

Context:
{context}
"""
    out = llm.invoke(prompt).content.strip()

    if word_count(out) > 500:
        out = " ".join(out.split()[:500])

    return out

def show_ranked_urls_section(ranked):
    strong_count = len([x for x in ranked[:10] if x["score"] >= STRONG_THRESHOLD])

    st.subheader("Step 2: Top Wikipedia Pages (Ranked)")
    st.caption(f"Query used: {st.session_state.query_used}")
    st.caption(f"Report topic: {st.session_state.topic_for_report}")

    if strong_count >= 5:
        st.success("Found strong matches (based on relevance scoring)")
    elif strong_count == 0:
        st.warning("No strong matches found. Showing best-available pages with relevance scores so you can choose")
    else:
        st.warning(f"Only {strong_count} strong match(es) found. Showing best-available pages with scores so you can choose")

    st.markdown(
        f"""
**How to read this**
- Scores are **relevance out of 100** based on your query and the page content snippet
- **Strong match** = score ≥ {STRONG_THRESHOLD}/100
- Even if scores are low, these are the best pages available from Wikipedia retrieval
"""
    )

def render_step_tracker():
    stage = st.session_state.stage
    done1 = bool(st.session_state.industry.strip())
    done2 = bool(st.session_state.ranked_docs)
    done3 = bool(st.session_state.report)

    current = 1
    if stage in ["clarify_1", "clarify_2", "force_keywords", "step2_results", "reported"]:
        current = 2
    if stage == "reported":
        current = 3

    def mark(done, stepnum):
        if done and stepnum < current:
            return "✅"
        if stepnum == current:
            return "➡️"
        return "⬜"

    st.markdown(
        f"""
### Progress
- {mark(done1, 1)} **Step 1** Input
- {mark(done2, 2)} **Step 2** Find relevant URLs
- {mark(done3, 3)} **Step 3** Generate report
"""
    )
    st.divider()

def show_banner_if_any():
    if st.session_state.banner_text:
        kind = st.session_state.banner_kind or "info"
        msg = st.session_state.banner_text

        if kind == "warning":
            st.warning(msg)
        elif kind == "error":
            st.error(msg)
        elif kind == "success":
            st.success(msg)
        else:
            st.info(msg)

        st.session_state.banner_text = ""
        st.session_state.banner_kind = ""

def run_retrieval_and_ranking(query: str, force_show_results: bool = False):
    with st.spinner("Retrieving Wikipedia pages..."):
        docs = retrieve_docs(query)

    with st.spinner("Ranking relevance (0–100)..."):
        ranked = rank_docs(query, docs)

    st.session_state.docs = docs
    st.session_state.ranked_docs = ranked
    st.session_state.query_used = query

    if force_show_results or st.session_state.clarify_round >= 2:
        st.session_state.confidence_mode = "needs_help"
        st.session_state.stage = "step2_results"
        st.rerun()

    if is_confident(ranked):
        st.session_state.confidence_mode = "confident"
        st.session_state.stage = "step2_results"
    else:
        st.session_state.confidence_mode = "needs_help"
        if st.session_state.clarify_round == 1:
            st.session_state.banner_text = "Couldn’t find strong matches even after your clarification. Try one of the clickable keywords below to guide the search"
            st.session_state.banner_kind = "warning"
            st.session_state.stage = "clarify_2"
        elif st.session_state.clarify_round == 0:
            st.session_state.stage = "clarify_1"
        else:
            st.session_state.stage = "force_keywords"

    st.rerun()

# ==========================================================
# ALWAYS SHOW STEP TRACKER + BANNER
# ==========================================================
render_step_tracker()
show_banner_if_any()

# ==========================================================
# UX: KEEP COMPLETED STEPS VISIBLE
# - Step 1 summary stays visible after user proceeds
# - Step 2 clarification summaries stay visible too
# ==========================================================
with st.expander("Step 1: Input (always visible)", expanded=True):
    st.subheader("Step 1: Input")
    industry_input = st.text_input(
        "Input the industry that you want to do market research",
        value=st.session_state.industry,
        placeholder="Example: pet market, electric vehicles, cosmetics in Vietnam",
        key="industry_input_main",
    )

    c1, c2 = st.columns(2)
    with c1:
        start_btn = st.button("Find relevant Wikipedia pages", type="primary", use_container_width=True, key="start_btn")
    with c2:
        clear_btn = st.button("Clear", use_container_width=True, key="clear_btn")

    if clear_btn:
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()

    if start_btn:
        if not industry_input.strip():
            st.warning("Please enter an industry")
            st.stop()

        st.session_state.industry = industry_input.strip()
        st.session_state.clarify_round = 0
        st.session_state.report = ""
        st.session_state.ranked_docs = []
        st.session_state.docs = []

        st.session_state.context_ideas = suggest_context_ideas(st.session_state.industry)

        # NEW: reset “latest intent”
        st.session_state.force_keyword_selected = ""
        st.session_state.force_keyword_text = ""
        st.session_state.topic_for_report = compute_topic_for_report(
            st.session_state.industry,
            st.session_state.clarification_1,
            st.session_state.clarification_2,
            forced_kw=""
        )

        query = build_query(st.session_state.industry)
        run_retrieval_and_ranking(query)

# ==========================================================
# STEP 2 PATH 2.1 — CLARIFICATION ROUND 1 (ALWAYS VISIBLE)
# ==========================================================
with st.expander("Step 2: Clarification (always visible when needed)", expanded=(st.session_state.stage in ["clarify_1", "clarify_2", "force_keywords"])):
    if st.session_state.stage == "clarify_1":
        st.subheader("Step 2: Clarification Needed (Round 1)")
        st.warning("I’m not confident I can pull the right pages yet. Add more detail so I don’t retrieve the wrong URLs")

        if st.session_state.context_ideas:
            st.markdown("Pick a context idea (optional)")
            cols = st.columns(len(st.session_state.context_ideas))
            for i, idea in enumerate(st.session_state.context_ideas):
                if cols[i].button(idea, use_container_width=True, key=f"idea1_{i}"):
                    st.session_state.clarification_1 = (
                        f"{st.session_state.clarification_1}, {idea}".strip(", ")
                        if st.session_state.clarification_1 else idea
                    )
                    # NEW: update topic intent immediately
                    st.session_state.topic_for_report = compute_topic_for_report(
                        st.session_state.industry,
                        st.session_state.clarification_1,
                        st.session_state.clarification_2,
                        st.session_state.force_keyword_selected
                    )
                    st.rerun()

        st.session_state.clarification_1 = st.text_input(
            "Add context (country, segment, B2B/B2C, product category, customer type)",
            value=st.session_state.clarification_1,
            placeholder="Example: UK, premium segment, B2C",
            key="clar1_box"
        )

        if st.button("Retry with clarification", type="primary", key="retry_clar1"):
            st.session_state.clarify_round = 1

            # NEW: update topic intent before retrieval
            st.session_state.topic_for_report = compute_topic_for_report(
                st.session_state.industry,
                st.session_state.clarification_1,
                st.session_state.clarification_2,
                st.session_state.force_keyword_selected
            )

            query = build_query(st.session_state.industry, st.session_state.clarification_1)
            run_retrieval_and_ranking(query)

    if st.session_state.stage == "clarify_2":
        st.subheader("Step 2: Clarification Needed (Round 2)")
        st.warning("Still not confident. Add one more layer of detail. After this, I will show best-available ranked pages even if matches are weak")

        if not st.session_state.context_ideas:
            st.session_state.context_ideas = suggest_context_ideas(st.session_state.industry)

        st.markdown("If you're stuck, click a suggestion to help you think")
        ideas = st.session_state.context_ideas[:6]
        if ideas:
            cols = st.columns(len(ideas))
            for i, idea in enumerate(ideas):
                if cols[i].button(idea, use_container_width=True, key=f"idea2_{i}"):
                    st.session_state.clarification_2 = (
                        f"{st.session_state.clarification_2}, {idea}".strip(", ")
                        if st.session_state.clarification_2 else idea
                    )
                    st.session_state.topic_for_report = compute_topic_for_report(
                        st.session_state.industry,
                        st.session_state.clarification_1,
                        st.session_state.clarification_2,
                        st.session_state.force_keyword_selected
                    )
                    st.rerun()

        st.session_state.clarification_2 = st.text_input(
            "Add more context (regulation focus, geography, customer segment, subcategory)",
            value=st.session_state.clarification_2,
            placeholder="Example: companionship animals, retail segment, supply chain focus",
            key="clar2_box"
        )

        if st.button("Retry again", type="primary", key="retry_clar2"):
            st.session_state.clarify_round = 2

            st.session_state.topic_for_report = compute_topic_for_report(
                st.session_state.industry,
                st.session_state.clarification_1,
                st.session_state.clarification_2,
                st.session_state.force_keyword_selected
            )

            query = build_query(st.session_state.industry, st.session_state.clarification_1, st.session_state.clarification_2)
            run_retrieval_and_ranking(query, force_show_results=True)

# ==========================================================
# STEP 2 PATH 2.3 — FORCE KEYWORDS (CLICK OR TYPE)
# ==========================================================
def render_force_keyword_panel():
    st.markdown("### Refine with a forcing keyword (optional)")
    st.caption("Click a chip to run a focused search, or type your own. This panel stays visible so you remember what you chose")

    if not st.session_state.force_keywords:
        st.session_state.force_keywords = suggest_force_keywords(
            st.session_state.industry,
            st.session_state.clarification_1,
            st.session_state.clarification_2
        )

    if st.session_state.force_keywords:
        cols = st.columns(4)
        for idx, kw in enumerate(st.session_state.force_keywords):
            col = cols[idx % 4]
            if col.button(kw, use_container_width=True, key=f"kwchip_{idx}"):
                st.session_state.force_keyword_selected = kw

                # NEW: topic intent updates to the most recent refined choice (this is what you asked)
                st.session_state.topic_for_report = compute_topic_for_report(
                    st.session_state.industry,
                    st.session_state.clarification_1,
                    st.session_state.clarification_2,
                    forced_kw=kw
                )

                query = build_query(
                    st.session_state.industry,
                    st.session_state.clarification_1,
                    st.session_state.clarification_2,
                    forced_kw=kw
                )
                run_retrieval_and_ranking(query, force_show_results=True)

    st.session_state.force_keyword_text = st.text_input(
        "Or type a forcing keyword",
        value=st.session_state.force_keyword_text,
        placeholder="Example: pet companionship, retail, regulation, consumer behaviour",
        key="force_kw_text"
    )

    if st.button("Run forced search", type="primary", key="run_forced_btn"):
        kw = st.session_state.force_keyword_text.strip()
        if not kw:
            st.warning("Type a keyword or click a chip")
            st.stop()
        st.session_state.force_keyword_selected = kw

        # NEW: topic intent becomes the most recent user-provided forcing keyword
        st.session_state.topic_for_report = compute_topic_for_report(
            st.session_state.industry,
            st.session_state.clarification_1,
            st.session_state.clarification_2,
            forced_kw=kw
        )

        query = build_query(
            st.session_state.industry,
            st.session_state.clarification_1,
            st.session_state.clarification_2,
            forced_kw=kw
        )
        run_retrieval_and_ranking(query, force_show_results=True)

with st.expander("Step 2: Forced keywords (always visible when needed)", expanded=(st.session_state.stage in ["force_keywords", "step2_results"] and st.session_state.confidence_mode == "needs_help")):
    if st.session_state.stage == "force_keywords":
        st.subheader("Step 2: Forced Search (System still unsure)")
        st.warning("Even after clarification, the request is still ambiguous for Wikipedia. Choose a forcing keyword to guide retrieval")
    render_force_keyword_panel()
    if st.session_state.force_keyword_selected:
        st.info(f"Forced keyword chosen: {st.session_state.force_keyword_selected}")
    if st.session_state.topic_for_report:
        st.caption(f"Current report topic: {st.session_state.topic_for_report}")

# ==========================================================
# STEP 2 RESULTS — SHOW URLS + SCORES (ALWAYS VISIBLE)
# ==========================================================
with st.expander("Step 2: Ranked URLs (always visible after retrieval)", expanded=bool(st.session_state.ranked_docs)):
    if st.session_state.ranked_docs:
        ranked = st.session_state.ranked_docs
        show_ranked_urls_section(ranked)

        top5 = ranked[:5]
        strong_in_top5 = [x for x in top5 if x["score"] >= STRONG_THRESHOLD]
        backup_in_top5 = [x for x in top5 if x["score"] < STRONG_THRESHOLD]

        st.markdown("#### Ranked results (top 5)")
        if strong_in_top5:
            st.markdown("**Strong matches**")
            for i, x in enumerate(strong_in_top5, 1):
                st.markdown(f"**{i}. {x['title']}**")
                st.caption(f"Relevance: {x['score']}/100")
                st.markdown(x["url"] if x["url"] else "_No URL available_")
                st.markdown("")

        if backup_in_top5:
            st.markdown("**Backup matches (less direct)**")
            for i, x in enumerate(backup_in_top5, 1):
                st.markdown(f"**{i}. {x['title']}**")
                st.caption(f"Relevance: {x['score']}/100")
                st.markdown(x["url"] if x["url"] else "_No URL available_")
                st.markdown("")

    else:
        st.caption("No ranked URLs yet. Complete Step 1 to retrieve Wikipedia pages")

# ==========================================================
# STEP 3 — REPORT (ALWAYS VISIBLE AFTER URLS EXIST)
# ==========================================================
with st.expander("Step 3: Report (always visible after Step 2)", expanded=(st.session_state.stage in ["step2_results", "reported"] and bool(st.session_state.ranked_docs))):
    if not st.session_state.ranked_docs:
        st.caption("Report generation will appear here after Step 2 retrieves ranked URLs")
    else:
        st.subheader("Step 3: Report")
        st.caption(f"Report topic used: {st.session_state.topic_for_report or st.session_state.industry}")

        ranked = st.session_state.ranked_docs
        top5 = ranked[:5]

        require_selection = (st.session_state.clarify_round >= 2)

        if not require_selection:
            strong_all = [x for x in ranked if x["score"] >= STRONG_THRESHOLD]
            auto_ok = len(strong_all) >= MIN_STRONG_FOR_AUTO

            if auto_ok:
                st.success("5 strong matches found, no need to select pages")
                if st.button("Generate report", type="primary", key="gen_report_auto"):
                    docs_for_report = [x["doc"] for x in ranked[:5]]
                    topic = st.session_state.topic_for_report or st.session_state.industry
                    st.session_state.report = generate_report(topic, docs_for_report)
                    st.session_state.stage = "reported"
                    st.rerun()
            else:
                require_selection = True

        if require_selection:
            st.info("Select which pages to use for the report (recommended: choose the highest scores first)")
            selected_docs = []
            for x in top5:
                label = f"{x['title']} ({x['score']}/100)"
                default_on = x["score"] >= STRONG_THRESHOLD
                if st.checkbox(label, value=default_on, key=f"sel_{x['title']}_{x['score']}"):
                    selected_docs.append(x["doc"])

            if st.button("Generate report", type="primary", key="gen_report_selected"):
                if not selected_docs:
                    st.warning("Select at least one page")
                    st.stop()
                topic = st.session_state.topic_for_report or st.session_state.industry
                st.session_state.report = generate_report(topic, selected_docs)
                st.session_state.stage = "reported"
                st.rerun()

# ==========================================================
# REPORT OUTPUT (ALWAYS VISIBLE ONCE GENERATED)
# ==========================================================
with st.expander("Output: Market Research Report (always visible once generated)", expanded=bool(st.session_state.report)):
    if st.session_state.report:
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
    else:
        st.caption("No report yet. Complete Step 3 to generate the report")

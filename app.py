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
    "ranked_docs": [],            # list of dicts: {score, doc, title, url}
    "confidence_mode": "unknown", # "confident" | "needs_help"
    "clarify_round": 0,           # 0 -> none, 1 -> asked once, 2 -> asked twice
    "context_ideas": [],          # clickable ideas to help user clarify
    "force_keywords": [],         # clickable chips for forcing
    "force_keyword_selected": "", # what user clicked
    "force_keyword_text": "",     # manual typed
    "query_used": "",             # final query for retrieval
    "report": "",
    "selection_required": False,  # whether user must pick pages for report
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("API Key & LLM")
api_key = st.sidebar.text_input("OpenAI API key", type="password", placeholder="sk-...")

# Only one model in dropdown (as your prof requires)
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
RETRIEVE_TOP_K = 10          # retrieve more than 5, then rank
STRONG_THRESHOLD = 70        # "highly relevant" threshold
MIN_STRONG_FOR_AUTO = 5      # if >= 5 strong pages, no selection needed

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
Example format:
UK B2C, Thailand B2B, Skincare only, EV charging, Premium segment, Regulations focus
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
- Should be Wikipedia-search friendly
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

def generate_report(industry: str, docs_for_report):
    context = "\n\n".join([(d.page_content or "")[:2000] for d in docs_for_report]).strip()

    prompt = f"""
You are a professional market research analyst writing a clean, readable Markdown brief.

Hard rules:
- Use ONLY the context below
- Do NOT invent facts, numbers, competitors, or claims not supported by the context
- STRICTLY under 500 words (target 420–480)
- Do NOT put headings and normal text on the same line
- Use headings + short paragraphs + bullets
- No separators like "---"

Output format (follow exactly):

## Industry Brief: {industry}

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

    # Enforce <= 500 words
    if word_count(out) > 500:
        out = " ".join(out.split()[:500])

    return out

def show_ranked_urls_section(ranked):
    strong = [x for x in ranked if x["score"] >= STRONG_THRESHOLD]
    top5 = ranked[:5]

    st.subheader("Step 2: Top Wikipedia Pages (Ranked)")

    if len(strong) >= 5:
        st.success("Found 5 highly relevant pages")
        st.session_state.selection_required = False
    else:
        st.warning(f"Only {len(strong)} highly relevant page(s) found")
        st.session_state.selection_required = True

    # explain grouping
    if len(strong) < 5:
        st.markdown(
            f"""
**What you’re seeing**
- **Strong matches (score ≥ {STRONG_THRESHOLD}/100):** the best pages the system could confidently link to your topic
- **Backup matches (score < {STRONG_THRESHOLD}/100):** still related, but less specific or less direct
"""
        )

    return top5, strong

def run_retrieval_and_ranking(query: str):
    with st.spinner("Retrieving Wikipedia pages..."):
        docs = retrieve_docs(query)

    with st.spinner("Ranking relevance (0–100)..."):
        ranked = rank_docs(query, docs)

    st.session_state.docs = docs
    st.session_state.ranked_docs = ranked
    st.session_state.query_used = query

    if is_confident(ranked):
        st.session_state.confidence_mode = "confident"
        st.session_state.stage = "step2_results"
    else:
        st.session_state.confidence_mode = "needs_help"
        # move to clarification round flow
        if st.session_state.clarify_round == 0:
            st.session_state.stage = "clarify_1"
        elif st.session_state.clarify_round == 1:
            st.session_state.stage = "clarify_2"
        else:
            st.session_state.stage = "force_keywords"

    st.rerun()

# ==========================================================
# STEP 1 — USER INPUT
# ==========================================================
st.subheader("Step 1: Input")
industry_input = st.text_input(
    "Input the industry that you want to do market research",
    value=st.session_state.industry,
    placeholder="Example: pet food market, electric vehicles, cosmetics in Vietnam",
)

c1, c2 = st.columns(2)
with c1:
    start_btn = st.button("Find relevant Wikipedia pages", type="primary", use_container_width=True)
with c2:
    clear_btn = st.button("Clear", use_container_width=True)

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
    st.session_state.context_ideas = suggest_context_ideas(st.session_state.industry)

    query = build_query(st.session_state.industry)
    run_retrieval_and_ranking(query)

# ==========================================================
# STEP 2 PATH 2.1 — CLARIFICATION ROUND 1
# ==========================================================
if st.session_state.stage == "clarify_1":
    st.subheader("Step 2: Clarification Needed (Round 1)")
    st.warning("I’m not confident I can pull the right pages yet. Add more detail so I don’t retrieve the wrong URLs")

    # show helpful context ideas (clickable)
    if st.session_state.context_ideas:
        st.markdown("Pick a context idea (optional)")
        cols = st.columns(len(st.session_state.context_ideas))
        for i, idea in enumerate(st.session_state.context_ideas):
            if cols[i].button(idea, use_container_width=True, key=f"idea1_{i}"):
                # append into clarification box
                if st.session_state.clarification_1:
                    st.session_state.clarification_1 = f"{st.session_state.clarification_1}, {idea}"
                else:
                    st.session_state.clarification_1 = idea
                st.rerun()

    st.session_state.clarification_1 = st.text_input(
        "Add context (country, segment, B2B/B2C, product category, customer type)",
        value=st.session_state.clarification_1,
        placeholder="Example: UK, pet food, premium segment, B2C",
    )

    if st.button("Retry with clarification", type="primary"):
        st.session_state.clarify_round = 1
        query = build_query(st.session_state.industry, st.session_state.clarification_1)
        run_retrieval_and_ranking(query)

# ==========================================================
# STEP 2 PATH 2.2 — CLARIFICATION ROUND 2
# ==========================================================
if st.session_state.stage == "clarify_2":
    st.subheader("Step 2: Clarification Needed (Round 2)")
    st.warning("Still not confident. Add one more layer of detail (this helps a lot for Wikipedia matching)")

    # generate fresh context ideas based on clarification_1
    # (kept simple: reuse existing ideas + add more if empty)
    if not st.session_state.context_ideas:
        st.session_state.context_ideas = suggest_context_ideas(st.session_state.industry)

    st.markdown("Optional: click a suggestion to add it")
    # show 6 chips max
    ideas = st.session_state.context_ideas[:6]
    cols = st.columns(len(ideas))
    for i, idea in enumerate(ideas):
        if cols[i].button(idea, use_container_width=True, key=f"idea2_{i}"):
            if st.session_state.clarification_2:
                st.session_state.clarification_2 = f"{st.session_state.clarification_2}, {idea}"
            else:
                st.session_state.clarification_2 = idea
            st.rerun()

    st.session_state.clarification_2 = st.text_input(
        "Add more context (regulation focus, geography, customer segment, subcategory)",
        value=st.session_state.clarification_2,
        placeholder="Example: franchise restaurants, Google reviews, customer experience",
    )

    if st.button("Retry again", type="primary"):
        st.session_state.clarify_round = 2
        query = build_query(st.session_state.industry, st.session_state.clarification_1, st.session_state.clarification_2)
        run_retrieval_and_ranking(query)

# ==========================================================
# STEP 2 PATH 2.3 — FORCE KEYWORDS (CLICK OR TYPE)
# chips stay visible even after results
# ==========================================================
def render_force_keyword_panel():
    st.markdown("### If still ambiguous: choose a forcing keyword")
    st.caption("Click a chip to run a focused search, or type your own. This panel stays visible so you remember what you chose")

    if not st.session_state.force_keywords:
        st.session_state.force_keywords = suggest_force_keywords(
            st.session_state.industry,
            st.session_state.clarification_1,
            st.session_state.clarification_2
        )

    # clickable chips (keep visible)
    if st.session_state.force_keywords:
        cols = st.columns(4)
        for idx, kw in enumerate(st.session_state.force_keywords):
            col = cols[idx % 4]
            if col.button(kw, use_container_width=True, key=f"kwchip_{idx}"):
                # clicking chip triggers retrieval+ranking and then shows urls with score
                st.session_state.force_keyword_selected = kw
                query = build_query(
                    st.session_state.industry,
                    st.session_state.clarification_1,
                    st.session_state.clarification_2,
                    forced_kw=kw
                )
                run_retrieval_and_ranking(query)

    st.session_state.force_keyword_text = st.text_input(
        "Or type a forcing keyword",
        value=st.session_state.force_keyword_text,
        placeholder="Example: pet food, pet industry, veterinary medicine, animal welfare",
    )

    if st.button("Run forced search", type="primary"):
        kw = st.session_state.force_keyword_text.strip()
        if not kw:
            st.warning("Type a keyword or click a chip")
            st.stop()
        st.session_state.force_keyword_selected = kw
        query = build_query(
            st.session_state.industry,
            st.session_state.clarification_1,
            st.session_state.clarification_2,
            forced_kw=kw
        )
        run_retrieval_and_ranking(query)

if st.session_state.stage == "force_keywords":
    st.subheader("Step 2: Forced Search (System still unsure)")
    st.warning("Even after clarification, the request is still ambiguous for Wikipedia. Choose a forcing keyword to guide retrieval")
    render_force_keyword_panel()

# ==========================================================
# STEP 2 RESULTS — SHOW URLS + SCORES
# always keep refinement panel visible when not confident
# ==========================================================
if st.session_state.stage == "step2_results":
    ranked = st.session_state.ranked_docs or []
    if not ranked:
        st.error("No results to show. Please try again")
        st.stop()

    # If we were in non-confident mode, keep force panel visible (smooth UX)
    if st.session_state.confidence_mode == "needs_help":
        with st.expander("Refine search (optional)", expanded=True):
            render_force_keyword_panel()
        if st.session_state.force_keyword_selected:
            st.info(f"Forced keyword used: {st.session_state.force_keyword_selected}")

    top5, strong = show_ranked_urls_section(ranked)

    # Display top results with score
    st.markdown("---")
    st.markdown("#### Ranked results")

    # Prepare lists (best effort)
    strong_list = [x for x in top5 if x["score"] >= STRONG_THRESHOLD]
    backup_list = [x for x in top5 if x["score"] < STRONG_THRESHOLD]

    if strong_list:
        st.markdown("**Strong matches**")
        for i, x in enumerate(strong_list, 1):
            st.markdown(f"**{i}. {x['title']}**")
            st.caption(f"Relevance: {x['score']}/100")
            st.markdown(x["url"] if x["url"] else "_No URL available_")
            st.markdown("")

    if backup_list:
        st.markdown("**Backup matches (less direct)**")
        for i, x in enumerate(backup_list, 1):
            st.markdown(f"**{i}. {x['title']}**")
            st.caption(f"Relevance: {x['score']}/100")
            st.markdown(x["url"] if x["url"] else "_No URL available_")
            st.markdown("")

    # Decide report flow
    st.subheader("Step 3: Report")

    # If we have 5 strong matches, auto-use top5 and generate report
    strong_in_all = [x for x in ranked if x["score"] >= STRONG_THRESHOLD]
    auto_ok = len(strong_in_all) >= MIN_STRONG_FOR_AUTO

    if auto_ok:
        st.success("5 strong matches found, no need to select pages")
        if st.button("Generate report", type="primary"):
            docs_for_report = [x["doc"] for x in ranked[:5]]
            st.session_state.report = generate_report(st.session_state.industry, docs_for_report)
            st.session_state.stage = "reported"
            st.rerun()
    else:
        st.info("Select which pages to use for the report (recommended: pick the strong ones first)")

        selected = []
        for x in top5:
            label = f"{x['title']} ({x['score']}/100)"
            if st.checkbox(label, value=(x["score"] >= STRONG_THRESHOLD), key=f"sel_{x['title']}_{x['score']}"):
                selected.append(x["doc"])

        if st.button("Generate report", type="primary"):
            if not selected:
                st.warning("Select at least one page")
                st.stop()
            st.session_state.report = generate_report(st.session_state.industry, selected)
            st.session_state.stage = "reported"
            st.rerun()

# ==========================================================
# STEP 3 OUTPUT — REPORT
# ==========================================================
if st.session_state.stage == "reported":
    st.subheader("Step 3: Market Research Report")
    st.markdown(st.session_state.report)

    wc = word_count(st.session_state.report)
    st.caption(f"Word count: {wc} / 500")

    st.download_button(
        "Download report (TXT)",
        data=st.session_state.report,
        file_name="industry_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

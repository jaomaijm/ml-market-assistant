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
    "Step-by-step Wikipedia-based market research report generator (≤ 500 words). "
    "If the request is ambiguous, the app will guide you to refine it before writing."
)

# ==========================================================
# SESSION STATE
# ==========================================================
defaults = {
    # Wizard step
    "step": 1,  # 1 input -> 2 find urls -> 3 choose urls (if needed) -> 4 report

    # Inputs
    "industry": "",
    "clarification": "",            # round 1 clarification
    "extra_context": "",            # round 2 clarification (optional)

    # Intent refiners (chips + typed)
    "forced_kw": "",
    "forced_kw_text": "",

    # Generated helpers
    "context_suggestions": [],
    "forced_kw_suggestions": [],

    # Retrieval
    "query_used": "",
    "ranked_docs": [],              # list of dicts {score,title,url,doc}
    "strong_count": 0,
    "top5_strong": 0,
    "retrieval_round": 0,           # 0 base, 1 clarification, 2 extra_context, 3 forced

    # Selection + report
    "need_selection": False,
    "selected_titles": set(),
    "topic_for_report": "",
    "report": "",

    # UX messaging
    "banner_kind": "",
    "banner_text": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
RETRIEVE_TOP_K = 10
STRONG_THRESHOLD = 70

# If we have at least this many strong pages, no selection needed
AUTO_STRONG_MIN = 5

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
    snippet = (doc.page_content or "")[:1400]
    judge_prompt = f"""
Score relevance from 0 to 100.

User query:
{query}

Wikipedia page title:
{title}

Wikipedia content snippet:
{snippet}

Return ONLY one integer 0-100.
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

def build_query(industry: str, clar: str = "", extra: str = "", forced_kw: str = "") -> str:
    parts = [industry.strip()]
    if clar.strip():
        parts.append(clar.strip())
    if extra.strip():
        parts.append(extra.strip())
    if forced_kw.strip():
        parts.append(forced_kw.strip())
    return " ".join(parts).strip()

def compute_topic_for_report(industry: str, clar: str, extra: str, forced_kw: str) -> str:
    # Always reflect the latest refined user intent (forced keyword has highest priority)
    bits = [industry.strip()]
    if forced_kw.strip():
        bits.append(f"focus: {forced_kw.strip()}")
    if clar.strip():
        bits.append(f"context: {clar.strip()}")
    if extra.strip():
        bits.append(f"more context: {extra.strip()}")
    return " | ".join(bits).strip()

def suggest_context_suggestions(industry: str):
    prompt = f"""
Give 6 short context ideas a user can click to clarify a market research request.
Each idea under 6 words.
Industry term: {industry}
Return ONLY as a comma-separated list.
"""
    resp = llm.invoke(prompt).content
    ideas = [x.strip() for x in resp.split(",") if x.strip()]
    return ideas[:6] if ideas else []

def suggest_forced_keywords(industry: str, clar: str, extra: str):
    prompt = f"""
Generate 8 specific Wikipedia-search-friendly forcing keywords to disambiguate.
Base industry: {industry}
Clarification: {clar}
Extra context: {extra}
Rules:
- 2 to 5 words
- specific sub-segments or interpretations
Return ONLY as comma-separated list.
"""
    resp = llm.invoke(prompt).content
    kws = [x.strip() for x in resp.split(",") if x.strip()]
    return kws[:8] if kws else []

def generate_report(topic: str, docs_for_report):
    context = "\n\n".join([(d.page_content or "")[:2000] for d in docs_for_report]).strip()

    prompt = f"""
You are a professional market research analyst writing a clean, readable Markdown brief.

Topic (follow this intent exactly):
{topic}

Hard rules:
- Use ONLY the context below
- Do NOT invent facts, numbers, competitors, or claims not supported by the context
- STRICTLY under 500 words (target 420–480)
- Use headings + short paragraphs + bullets
- Headings on their own line
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

def run_search(query: str, round_label: str):
    with st.spinner(f"Searching Wikipedia ({round_label})..."):
        docs = retrieve_docs(query)
    with st.spinner("Scoring relevance (0–100)..."):
        ranked = rank_docs(query, docs)

    st.session_state.query_used = query
    st.session_state.ranked_docs = ranked

    top5 = ranked[:5]
    strong_all = [x for x in ranked if x["score"] >= STRONG_THRESHOLD]
    strong_top5 = [x for x in top5 if x["score"] >= STRONG_THRESHOLD]

    st.session_state.strong_count = len(strong_all)
    st.session_state.top5_strong = len(strong_top5)

    # Decide if user must choose URLs
    st.session_state.need_selection = not (len(strong_all) >= AUTO_STRONG_MIN)

def render_stepper():
    step = st.session_state.step
    labels = ["1 Input", "2 Find URLs", "3 Choose URLs", "4 Report"]
    done = [False, False, False, False]
    done[0] = bool(st.session_state.industry.strip())
    done[1] = bool(st.session_state.ranked_docs)
    done[2] = (not st.session_state.need_selection) or (len(st.session_state.selected_titles) > 0)
    done[3] = bool(st.session_state.report.strip())

    cols = st.columns(4)
    for i, lab in enumerate(labels, start=1):
        icon = "⬜"
        if done[i - 1] and i < step:
            icon = "✅"
        elif i == step:
            icon = "➡️"
        cols[i - 1].markdown(f"**{icon} {lab}**")
    st.divider()

def render_summary_bar():
    # Human-brain-friendly: always remind what the system currently understands
    industry = st.session_state.industry.strip() or "—"
    clar = st.session_state.clarification.strip() or "—"
    extra = st.session_state.extra_context.strip() or "—"
    forced = st.session_state.forced_kw.strip() or "—"
    topic = st.session_state.topic_for_report.strip() or "—"

    st.markdown("#### Current understanding")
    st.markdown(
        f"""
- **Industry**: {industry}  
- **Clarification**: {clar}  
- **More context**: {extra}  
- **Forced keyword**: {forced}  
- **Report topic**: {topic}
"""
    )
    st.divider()

def top5_display(ranked):
    top5 = ranked[:5]
    strong = [x for x in top5 if x["score"] >= STRONG_THRESHOLD]
    backup = [x for x in top5 if x["score"] < STRONG_THRESHOLD]

    # Clear, transparent labels (exactly what you asked)
    if len(strong) >= 5:
        st.success("Path 1: Confident — pulled 5 strong matches")
    elif len(strong) == 0:
        st.warning("Path 2: Not confident — no strong matches. Showing best available pages with relevance scores")
    else:
        st.warning(
            f"Path 2: Partially confident — only {len(strong)} strong match(es). "
            "Showing best available pages and weaker backups with scores"
        )

    st.caption("Scores are relevance out of 100. Strong match = 70+")
    st.markdown(f"Query used: `{st.session_state.query_used}`")

    if strong:
        st.markdown("### Strong matches")
        for i, x in enumerate(strong, 1):
            st.markdown(f"**{i}. {x['title']}**")
            st.caption(f"Relevance: {x['score']}/100")
            st.markdown(x["url"] or "_No URL available_")
            st.markdown("")

    if backup:
        st.markdown("### Backup matches (less direct)")
        for i, x in enumerate(backup, 1):
            st.markdown(f"**{i}. {x['title']}**")
            st.caption(f"Relevance: {x['score']}/100")
            st.markdown(x["url"] or "_No URL available_")
            st.markdown("")

# ==========================================================
# TOP UX: Stepper + banner + summary
# ==========================================================
render_stepper()
show_banner()
render_summary_bar()

# ==========================================================
# STEP 1: USER INPUT
# ==========================================================
st.subheader("Step 1: Input the industry you want to research")

industry_input = st.text_input(
    "Industry",
    value=st.session_state.industry,
    placeholder="Example: pet market, electric vehicles, cosmetics in Vietnam",
    key="industry_input",
)

c1, c2 = st.columns([1, 1])
with c1:
    start = st.button("Continue to Step 2", type="primary", use_container_width=True)
with c2:
    clear = st.button("Clear all", use_container_width=True)

if clear:
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

if start:
    if not industry_input.strip():
        st.warning("Please enter an industry")
        st.stop()

    # Reset downstream state but keep UX smooth
    st.session_state.industry = industry_input.strip()
    st.session_state.step = 2

    st.session_state.clarification = ""
    st.session_state.extra_context = ""
    st.session_state.forced_kw = ""
    st.session_state.forced_kw_text = ""
    st.session_state.selected_titles = set()
    st.session_state.report = ""
    st.session_state.ranked_docs = []
    st.session_state.retrieval_round = 0

    # Helpful suggestions ready for the brain (but not forcing)
    st.session_state.context_suggestions = suggest_context_suggestions(st.session_state.industry)
    st.session_state.forced_kw_suggestions = []

    st.session_state.topic_for_report = compute_topic_for_report(
        st.session_state.industry, "", "", ""
    )

    st.rerun()

st.divider()

# ==========================================================
# STEP 2: FIND URLS (with 2-stage clarification BEFORE forcing)
# ==========================================================
st.subheader("Step 2: Find 5 relevant Wikipedia pages (with relevance scores)")

if st.session_state.step < 2:
    st.info("Complete Step 1 first")
    st.stop()

# 2A) First attempt (base query) if we haven't searched yet
if not st.session_state.ranked_docs and st.session_state.retrieval_round == 0:
    query = build_query(st.session_state.industry)
    st.session_state.query_used = query
    run_search(query, "base query")
    st.session_state.retrieval_round = 1

    # If confident, show results and move forward automatically
    if st.session_state.strong_count >= AUTO_STRONG_MIN:
        st.session_state.banner_kind = "success"
        st.session_state.banner_text = "I’m confident these pages match your topic. Review Step 2 results below"
        st.session_state.need_selection = False
        st.session_state.step = 4  # skip Step 3 selection
    else:
        st.session_state.banner_kind = "warning"
        st.session_state.banner_text = "Not confident yet. Add a quick clarification so I don’t pull the wrong pages"
        st.session_state.step = 2
    st.rerun()

# Always show the ranked results from the latest search
if st.session_state.ranked_docs:
    top5_display(st.session_state.ranked_docs)

st.divider()

# 2B) Clarification Round 1 (only if not confident)
if st.session_state.ranked_docs and st.session_state.strong_count < AUTO_STRONG_MIN and st.session_state.retrieval_round == 1:
    st.markdown("### Step 2.1: Add clarification (Round 1)")
    st.caption("Pick a suggestion or type your own. This helps the system understand your intent before forcing keywords")

    # Clickable context ideas + typed box (both)
    if st.session_state.context_suggestions:
        cols = st.columns(len(st.session_state.context_suggestions))
        for i, idea in enumerate(st.session_state.context_suggestions):
            if cols[i].button(idea, use_container_width=True, key=f"ctx1_{i}"):
                st.session_state.clarification = (
                    f"{st.session_state.clarification}, {idea}".strip(", ")
                    if st.session_state.clarification else idea
                )
                st.session_state.topic_for_report = compute_topic_for_report(
                    st.session_state.industry,
                    st.session_state.clarification,
                    st.session_state.extra_context,
                    st.session_state.forced_kw
                )
                st.rerun()

    st.session_state.clarification = st.text_input(
        "Clarification",
        value=st.session_state.clarification,
        placeholder="Example: UK, companionship animals, B2C, retail, premium segment",
        key="clar1_input",
    )

    r1 = st.button("Search again with clarification", type="primary", use_container_width=True, key="run_r1")
    if r1:
        st.session_state.topic_for_report = compute_topic_for_report(
            st.session_state.industry,
            st.session_state.clarification,
            st.session_state.extra_context,
            st.session_state.forced_kw
        )
        query = build_query(st.session_state.industry, st.session_state.clarification)
        st.session_state.query_used = query
        run_search(query, "clarification round 1")
        st.session_state.retrieval_round = 2

        if st.session_state.strong_count >= AUTO_STRONG_MIN:
            st.session_state.banner_kind = "success"
            st.session_state.banner_text = "Now confident. Review the ranked pages below"
            st.session_state.need_selection = False
            st.session_state.step = 4
        else:
            st.session_state.banner_kind = "warning"
            st.session_state.banner_text = "Still not confident. Add one more detail, then I’ll offer forcing keywords"
            st.session_state.step = 2
        st.rerun()

st.divider()

# 2C) Clarification Round 2 (still before forcing)
if st.session_state.ranked_docs and st.session_state.strong_count < AUTO_STRONG_MIN and st.session_state.retrieval_round == 2:
    st.markdown("### Step 2.2: Add more context (Round 2)")
    st.caption("One more hint makes Wikipedia retrieval much cleaner. After this, you’ll get forcing keywords to click")

    # Give brain-friendly suggestions again
    if not st.session_state.context_suggestions:
        st.session_state.context_suggestions = suggest_context_suggestions(st.session_state.industry)

    if st.session_state.context_suggestions:
        cols = st.columns(len(st.session_state.context_suggestions))
        for i, idea in enumerate(st.session_state.context_suggestions):
            if cols[i].button(idea, use_container_width=True, key=f"ctx2_{i}"):
                st.session_state.extra_context = (
                    f"{st.session_state.extra_context}, {idea}".strip(", ")
                    if st.session_state.extra_context else idea
                )
                st.session_state.topic_for_report = compute_topic_for_report(
                    st.session_state.industry,
                    st.session_state.clarification,
                    st.session_state.extra_context,
                    st.session_state.forced_kw
                )
                st.rerun()

    st.session_state.extra_context = st.text_input(
        "More context",
        value=st.session_state.extra_context,
        placeholder="Example: adoption, pet insurance, vet services, supply chain, consumer behaviour",
        key="clar2_input",
    )

    r2 = st.button("Search again with more context", type="primary", use_container_width=True, key="run_r2")
    if r2:
        st.session_state.topic_for_report = compute_topic_for_report(
            st.session_state.industry,
            st.session_state.clarification,
            st.session_state.extra_context,
            st.session_state.forced_kw
        )
        query = build_query(st.session_state.industry, st.session_state.clarification, st.session_state.extra_context)
        st.session_state.query_used = query
        run_search(query, "clarification round 2")
        st.session_state.retrieval_round = 3

        # IMPORTANT CHANGE you asked:
        # even if still not confident, DO NOT block — show bar + forcing keywords next
        if st.session_state.strong_count >= AUTO_STRONG_MIN:
            st.session_state.banner_kind = "success"
            st.session_state.banner_text = "Now confident. Review the ranked pages below"
            st.session_state.need_selection = False
            st.session_state.step = 4
        else:
            st.session_state.banner_kind = "warning"
            st.session_state.banner_text = "Couldn’t find strong matches. Use forcing keywords below (click or type) to guide retrieval"
            st.session_state.step = 2
        st.rerun()

st.divider()

# 2D) Forcing keywords flow (click => run search => show URLs with score)
if st.session_state.ranked_docs and st.session_state.strong_count < AUTO_STRONG_MIN and st.session_state.retrieval_round >= 3:
    st.markdown("### Step 2.3: Forcing keywords (click or type)")
    st.caption("Click a keyword to run a focused search immediately. Or type your own then search")

    if not st.session_state.forced_kw_suggestions:
        st.session_state.forced_kw_suggestions = suggest_forced_keywords(
            st.session_state.industry,
            st.session_state.clarification,
            st.session_state.extra_context
        )

    # Clickable chips
    if st.session_state.forced_kw_suggestions:
        cols = st.columns(4)
        for idx, kw in enumerate(st.session_state.forced_kw_suggestions):
            col = cols[idx % 4]
            if col.button(kw, use_container_width=True, key=f"kw_{idx}"):
                st.session_state.forced_kw = kw
                st.session_state.forced_kw_text = ""
                st.session_state.topic_for_report = compute_topic_for_report(
                    st.session_state.industry,
                    st.session_state.clarification,
                    st.session_state.extra_context,
                    st.session_state.forced_kw
                )
                query = build_query(
                    st.session_state.industry,
                    st.session_state.clarification,
                    st.session_state.extra_context,
                    forced_kw=st.session_state.forced_kw
                )
                st.session_state.query_used = query
                run_search(query, "forced keyword (chip)")

                # Always show results after click (do not keep changing chips)
                st.session_state.banner_kind = "info"
                st.session_state.banner_text = "Forced search completed. Review the ranked pages above (Step 2 results)"
                st.rerun()

    # Typed forcing keyword
    st.session_state.forced_kw_text = st.text_input(
        "Type forcing keyword",
        value=st.session_state.forced_kw_text,
        placeholder="Example: pet companionship, pet industry, animal welfare",
        key="forced_kw_text_box"
    )

    typed_btn = st.button("Search with typed forcing keyword", type="primary", use_container_width=True, key="run_forced_typed")
    if typed_btn:
        kw = st.session_state.forced_kw_text.strip()
        if not kw:
            st.warning("Type a keyword or click a chip")
            st.stop()
        st.session_state.forced_kw = kw
        st.session_state.topic_for_report = compute_topic_for_report(
            st.session_state.industry,
            st.session_state.clarification,
            st.session_state.extra_context,
            st.session_state.forced_kw
        )
        query = build_query(
            st.session_state.industry,
            st.session_state.clarification,
            st.session_state.extra_context,
            forced_kw=st.session_state.forced_kw
        )
        st.session_state.query_used = query
        run_search(query, "forced keyword (typed)")

        st.session_state.banner_kind = "info"
        st.session_state.banner_text = "Forced search completed. Review the ranked pages above (Step 2 results)"
        st.rerun()

# Decide next step after we have ranked docs:
# - If confident => Step 4 report (no selection)
# - If not confident => Step 3 choose URLs
if st.session_state.ranked_docs:
    if st.session_state.strong_count >= AUTO_STRONG_MIN:
        st.session_state.need_selection = False
        st.session_state.step = max(st.session_state.step, 4)
    else:
        st.session_state.need_selection = True
        st.session_state.step = max(st.session_state.step, 3)

st.divider()

# ==========================================================
# STEP 3: USER CHOOSES URLS (only if not confident)
# ==========================================================
st.subheader("Step 3: Choose which pages to use for the report")

if st.session_state.step < 3:
    st.info("This step appears only when the system is not confident about 5 strong matches")
else:
    ranked = st.session_state.ranked_docs
    top5 = ranked[:5]

    # Explain exactly what’s happening (human brain friendly)
    strong_in_top5 = [x for x in top5 if x["score"] >= STRONG_THRESHOLD]
    backup_in_top5 = [x for x in top5 if x["score"] < STRONG_THRESHOLD]

    st.markdown(
        f"""
- **Strong pages found**: {len(strong_in_top5)} / 5  
- **Backup pages**: {len(backup_in_top5)} / 5  
Pick what you want included in the report. Recommended: select highest scores first
"""
    )

    # Default selection: all strong pages; if none strong, select top 3
    default_titles = set([x["title"] for x in strong_in_top5])
    if not default_titles:
        default_titles = set([x["title"] for x in top5[:3]])

    # Persist selection set
    if not st.session_state.selected_titles:
        st.session_state.selected_titles = set(default_titles)

    for x in top5:
        label = f"{x['title']} ({x['score']}/100)"
        checked = x["title"] in st.session_state.selected_titles
        new_val = st.checkbox(label, value=checked, key=f"pick_{x['title']}_{x['score']}")
        if new_val and x["title"] not in st.session_state.selected_titles:
            st.session_state.selected_titles.add(x["title"])
        if (not new_val) and x["title"] in st.session_state.selected_titles:
            st.session_state.selected_titles.remove(x["title"])

    st.caption("Your selected pages will be the only sources used to generate the report")

    if st.button("Continue to Step 4 (Generate report)", type="primary", use_container_width=True, key="to_step4"):
        st.session_state.step = 4
        st.rerun()

st.divider()

# ==========================================================
# STEP 4: REPORT
# ==========================================================
st.subheader("Step 4: Generate report (≤ 500 words)")

if st.session_state.step < 4:
    st.info("Complete Step 2 (and Step 3 if needed) first")
else:
    # Make sure topic always reflects the latest refined user intent
    st.session_state.topic_for_report = compute_topic_for_report(
        st.session_state.industry,
        st.session_state.clarification,
        st.session_state.extra_context,
        st.session_state.forced_kw
    )

    st.caption(f"Report topic: {st.session_state.topic_for_report}")

    docs_for_report = []
    ranked = st.session_state.ranked_docs
    if not ranked:
        st.warning("No pages available to write a report")
        st.stop()

    if st.session_state.need_selection:
        # Use user-selected docs (titles)
        selected = st.session_state.selected_titles
        docs_for_report = [x["doc"] for x in ranked[:5] if x["title"] in selected]
        if not docs_for_report:
            st.warning("Select at least one page in Step 3")
            st.stop()
    else:
        # Confident: use top 5
        docs_for_report = [x["doc"] for x in ranked[:5]]

    g1, g2 = st.columns([1, 1])
    with g1:
        gen = st.button("Generate report", type="primary", use_container_width=True, key="gen_report")
    with g2:
        regen = st.button("Regenerate", use_container_width=True, key="regen_report")

    if regen:
        st.session_state.report = ""
        st.rerun()

    if gen:
        with st.spinner("Writing report..."):
            st.session_state.report = generate_report(st.session_state.topic_for_report, docs_for_report)
        st.session_state.banner_kind = "success"
        st.session_state.banner_text = "Report generated"
        st.rerun()

# ==========================================================
# REPORT OUTPUT
# ==========================================================
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

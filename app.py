import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Market Research Assistant", page_icon="🔎", layout="wide")
st.title("Market Research Assistant")
st.caption(
    "Step 1: Enter industry → Step 2: Find most relevant Wikipedia pages → Step 3: Generate a clean report under 500 words"
)

# ==========================================================
# SESSION STATE
# ==========================================================
defaults = {
    "stage": "start",
    "industry": "",
    "docs": [],
    "ranked_docs": [],
    "clarification_round": 0,
    "clarification_text": "",
    "force_keyword_text": "",
    "force_keyword_selected": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("API Key")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to begin")
    st.stop()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    openai_api_key=api_key,
    temperature=0.2
)

TOP_K = 12  # retrieve a bit more, then rank down to 5

# ==========================================================
# HELPERS
# ==========================================================
def extract_url(doc):
    md = getattr(doc, "metadata", {}) or {}
    return md.get("source") or md.get("url") or ""

def extract_title(doc):
    md = getattr(doc, "metadata", {}) or {}
    return md.get("title") or "Wikipedia Page"

def retrieve_docs(query):
    retriever = WikipediaRetriever(top_k_results=TOP_K, lang="en")
    return retriever.invoke(query)

def safe_int_score(text, default=50):
    nums = re.findall(r"\d+", text or "")
    if not nums:
        return default
    score = int(nums[0])
    return max(0, min(100, score))

def rank_documents(industry, docs):
    ranked = []
    for d in docs:
        title = extract_title(d)
        preview = (d.page_content or "")[:900]

        prompt = f"""
Score the relevance of this Wikipedia page to the industry/topic: "{industry}"

Return ONLY one integer between 0 and 100
No other words

Title: {title}
Content preview: {preview}
""".strip()

        try:
            score_text = llm.invoke(prompt).content.strip()
            score = safe_int_score(score_text, default=50)
        except:
            score = 50

        ranked.append((score, d))

    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked

def word_count(text):
    return len(re.findall(r"\b\w+\b", text or ""))

def enforce_under_500(text):
    tokens = (text or "").split()
    return " ".join(tokens[:500])

def normalize_markdown(report: str) -> str:
    """
    Fix common LLM formatting problems:
    - headings + body on same line, e.g. "### Scope The industry..."
    - missing blank lines around headings/bullets
    """
    if not report:
        return report

    lines = report.splitlines()
    out = []

    # Expected section headings (so we can split "### Scope blah blah")
    expected_h3 = [
        "Scope",
        "Market Offering",
        "Value Chain",
        "Competitive Landscape",
        "Key Trends",
        "Limits",
    ]

    for line in lines:
        l = line.rstrip()

        # If heading and body got jammed together: "### Scope The..."
        for h in expected_h3:
            if l.startswith(f"### {h} "):
                # split into heading + body
                body = l[len(f"### {h} "):].strip()
                out.append(f"### {h}")
                out.append("")
                if body:
                    out.append(body)
                l = None
                break
        if l is None:
            continue

        # Ensure blank line after headings
        if re.match(r"^#{2,6}\s+\S", l):
            out.append(l)
            out.append("")
            continue

        out.append(l)

    cleaned = "\n".join(out)

    # Ensure a blank line before bullet lists for readability
    cleaned = re.sub(r"([^\n])\n(-\s)", r"\1\n\n\2", cleaned)

    # Remove excessive blank lines (keep max 2)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned

def suggest_clarifications(industry: str):
    prompt = f"""
User entered this industry/topic:
"{industry}"

Generate 6 short clarification options the user could click to narrow scope
Examples of what clarifications can be:
- country/region
- sub-segment (e.g., services vs product)
- customer type (B2B/B2C)
- price tier (mass vs premium)
- channel (online/offline)
- specific species/product category (if relevant)

Rules:
- Output ONLY a comma-separated list
- Each option must be short (3–8 words)
- No numbering, no extra text
""".strip()

    try:
        text = llm.invoke(prompt).content.strip()
        options = [o.strip() for o in text.split(",") if o.strip()]
        # keep unique and limit
        seen = set()
        uniq = []
        for o in options:
            key = o.lower()
            if key not in seen:
                uniq.append(o)
                seen.add(key)
        return uniq[:6] if uniq else []
    except:
        return []

def suggest_force_keywords(industry: str, clarification: str = ""):
    prompt = f"""
Base topic:
"{industry}"

Clarification (if any):
"{clarification}"

Generate 6 specific Wikipedia-friendly search keywords/phrases that would help retrieve the most relevant pages

Rules:
- Output ONLY a comma-separated list
- Each keyword 2–6 words
- No numbering, no extra text
""".strip()

    try:
        text = llm.invoke(prompt).content.strip()
        options = [o.strip() for o in text.split(",") if o.strip()]
        seen = set()
        uniq = []
        for o in options:
            key = o.lower()
            if key not in seen:
                uniq.append(o)
                seen.add(key)
        return uniq[:6] if uniq else []
    except:
        return []

# ==========================================================
# STEP 1 – USER INPUT
# ==========================================================
st.divider()
st.subheader("Step 1: Enter Industry")

industry_input = st.text_input(
    "Industry",
    placeholder="Example: Exotic pet care services",
    value=st.session_state.industry
)

if st.button("Find Relevant Pages", type="primary"):
    if not industry_input.strip():
        st.warning("Please enter an industry")
        st.stop()

    # reset
    st.session_state.industry = industry_input.strip()
    st.session_state.docs = []
    st.session_state.ranked_docs = []
    st.session_state.clarification_round = 0
    st.session_state.clarification_text = ""
    st.session_state.force_keyword_text = ""
    st.session_state.force_keyword_selected = ""
    st.session_state.stage = "retrieving"

    with st.spinner("Retrieving and ranking pages..."):
        docs = retrieve_docs(st.session_state.industry)
        ranked = rank_documents(st.session_state.industry, docs)

    st.session_state.ranked_docs = ranked

    top5 = ranked[:5]
    top_scores = [s for s, _ in top5]

    # High confidence: 5 pages all strong
    if len(top_scores) >= 5 and all(s >= 70 for s in top_scores):
        st.session_state.docs = [d for _, d in top5]
        st.session_state.stage = "ready"
    else:
        st.session_state.stage = "clarification"
        st.session_state.clarification_round = 1

    st.rerun()

# ==========================================================
# STEP 2 – PATH 2.1 Clarification Round 1 (with suggestion chips)
# ==========================================================
if st.session_state.stage == "clarification" and st.session_state.clarification_round == 1:
    st.subheader("Step 2: Clarification Needed")
    st.warning("Not fully confident the top pages match your intended scope")

    st.markdown("Pick a clarification (or type your own) to help the system avoid wrong URLs")

    suggestions = suggest_clarifications(st.session_state.industry)
    if suggestions:
        chip_cols = st.columns(min(6, len(suggestions)))
        for i, opt in enumerate(suggestions):
            if chip_cols[i % len(chip_cols)].button(opt):
                st.session_state.clarification_text = opt
                st.rerun()

    clarification = st.text_input(
        "Clarification",
        placeholder="Example: UK market, B2C, premium services",
        value=st.session_state.clarification_text,
        key="clarification_input"
    )

    if st.button("Refine Search", type="primary"):
        if not clarification.strip():
            st.warning("Please add a clarification or click a suggestion")
            st.stop()

        st.session_state.clarification_text = clarification.strip()
        refined_query = f"{st.session_state.industry} ({st.session_state.clarification_text})"

        with st.spinner("Re-ranking..."):
            docs = retrieve_docs(refined_query)
            ranked = rank_documents(refined_query, docs)

        st.session_state.ranked_docs = ranked
        top5 = ranked[:5]
        top_scores = [s for s, _ in top5]

        if len(top_scores) >= 5 and all(s >= 70 for s in top_scores):
            st.session_state.docs = [d for _, d in top5]
            st.session_state.stage = "ready"
        else:
            st.session_state.stage = "force_keywords"
            st.session_state.clarification_round = 2

        st.rerun()

# ==========================================================
# STEP 2 – PATH 2.3 Forced Keywords Round 2 (click OR type)
# ==========================================================
if st.session_state.stage == "force_keywords":
    st.subheader("Step 2: Choose a More Specific Keyword")
    st.warning("Still uncertain after clarification. Choose a more specific direction")

    force_suggestions = suggest_force_keywords(
        st.session_state.industry,
        st.session_state.clarification_text
    )

    st.markdown("Click a keyword or type your own")

    if force_suggestions:
        chip_cols = st.columns(min(6, len(force_suggestions)))
        for i, opt in enumerate(force_suggestions):
            if chip_cols[i % len(chip_cols)].button(opt):
                st.session_state.force_keyword_selected = opt
                st.session_state.force_keyword_text = opt
                st.rerun()

    forced = st.text_input(
        "Forced keyword",
        placeholder="Example: exotic pet boarding, reptile pet store",
        value=st.session_state.force_keyword_text,
        key="forced_keyword_input"
    )

    if st.button("Run Forced Search", type="primary"):
        chosen = forced.strip()
        if not chosen:
            st.warning("Pick a keyword or type one")
            st.stop()

        st.session_state.force_keyword_text = chosen

        final_query = f"{st.session_state.industry} ({st.session_state.clarification_text}) {chosen}".strip()

        with st.spinner("Final ranking..."):
            docs = retrieve_docs(final_query)
            ranked = rank_documents(final_query, docs)

        st.session_state.ranked_docs = ranked
        st.session_state.docs = [d for _, d in ranked[:5]]
        st.session_state.stage = "ready"
        st.rerun()

# ==========================================================
# STEP 2 – SHOW URLS WITH SCORES
# ==========================================================
if st.session_state.stage == "ready":
    st.subheader("Step 2: Top Retrieved Pages (Ranked)")

    ranked = st.session_state.ranked_docs
    if not ranked:
        st.info("No pages ranked yet")
    else:
        strong = [r for r in ranked if r[0] >= 70]
        if len(strong) < 5:
            st.info(
                f"Only {len(strong)} highly relevant pages found (score ≥ 70). "
                "The remaining pages are best-available matches from Wikipedia"
            )

        for score, doc in ranked[:5]:
            title = extract_title(doc)
            url = extract_url(doc)

            st.markdown(f"**{title}**")
            st.caption(f"Relevance score: {score}/100")
            if url:
                st.markdown(url)
            st.markdown("---")

# ==========================================================
# STEP 3 – REPORT
# ==========================================================
if st.session_state.stage == "ready" and st.session_state.docs:
    st.subheader("Step 3: Generate Report")

    if st.button("Generate Report", type="primary"):
        context = "\n\n".join([(d.page_content or "") for d in st.session_state.docs]).strip()

        prompt = f"""
You are a professional market research analyst.

Hard rules:
- Use ONLY the Wikipedia context provided
- Do NOT invent facts
- STRICTLY under 500 words
- Output clean Markdown ONLY
- Every heading must be on its own line
- Put a blank line after each heading
- Use bullets where appropriate
- Do NOT write headings and body text on the same line

Output format exactly:

## Industry Brief: {st.session_state.industry}

### Scope
(1–2 sentences)

### Market Offering
- bullet
- bullet

### Value Chain
- bullet
- bullet

### Competitive Landscape
- bullet
- bullet

### Key Trends
- bullet
- bullet

### Limits
(1–2 sentences)

Wikipedia Context:
{context}
""".strip()

        with st.spinner("Generating report..."):
            raw_report = llm.invoke(prompt).content.strip()

        raw_report = enforce_under_500(raw_report)
        cleaned_report = normalize_markdown(raw_report)

        st.markdown("## Industry Report")
        st.markdown(cleaned_report)
        st.caption(f"Word count: {word_count(cleaned_report)} / 500")

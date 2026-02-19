import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Market Research Assistant", page_icon="🔎", layout="wide")
st.title("Market Research Assistant")

# ==========================================================
# SESSION STATE DEFAULTS
# ==========================================================
defaults = {
    "stage": "start",
    "industry": "",
    "clarification_text": "",
    "force_keyword_text": "",
    "force_keyword_selected": "",
    "docs": [],
    "ranked_docs": [],
    "report": "",
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
    st.info("Enter your OpenAI API key to begin")
    st.stop()

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2, openai_api_key=api_key)

TOP_K = 8  # retrieve more first for ranking

# ==========================================================
# HELPERS
# ==========================================================
def retrieve_docs(query):
    retriever = WikipediaRetriever(top_k_results=TOP_K, lang="en")
    return retriever.invoke(query)

def extract_title(doc):
    return doc.metadata.get("title", "Wikipedia page")

def extract_url(doc):
    return doc.metadata.get("source", "")

def word_count(text):
    return len(re.findall(r"\b\w+\b", text))

def rank_documents(query, docs):
    ranked = []
    for d in docs:
        score_prompt = f"""
Score relevance (0-100) of this page for this industry.

Industry: {query}

Page Title: {extract_title(d)}
Content Summary:
{d.page_content[:1500]}

Return ONLY a number.
"""
        try:
            score = int(re.findall(r"\d+", llm.invoke(score_prompt).content)[0])
        except:
            score = 0
        ranked.append((score, d))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked

def assess_confidence(ranked_docs):
    strong = [s for s, _ in ranked_docs if s >= 70]
    return len(strong) >= 5

def suggest_force_keywords(industry, clarification):
    prompt = f"""
Suggest 5 more specific sub-segment keywords for this industry.

Industry: {industry}
Clarification: {clarification}

Return as comma-separated list only.
"""
    try:
        resp = llm.invoke(prompt).content
        return [x.strip() for x in resp.split(",") if x.strip()]
    except:
        return []

def generate_report(industry, docs):
    context = "\n\n".join([d.page_content[:2000] for d in docs])

    prompt = f"""
You are a professional market research analyst.

STRICT RULES:
- Use ONLY the context below
- Do NOT invent information
- Under 500 words
- Proper markdown formatting
- Use headings and normal paragraph text correctly

Output EXACT structure:

## Industry Brief: {industry}

### Scope
(1-2 sentences normal paragraph)

### Market Offering
- bullet
- bullet

### Value Chain
- bullet
- bullet

### Competitive Landscape
- bullet
- bullet

### Key Trends and Drivers
- bullet
- bullet

### Limits
(1-2 sentences normal paragraph)

Context:
{context}
"""

    report = llm.invoke(prompt).content.strip()

    if word_count(report) > 500:
        report = " ".join(report.split()[:500])

    return report

# ==========================================================
# STEP 1 — USER INPUT
# ==========================================================
st.subheader("Step 1: Enter Industry")

industry = st.text_input("Industry")

if st.button("Search"):
    if not industry.strip():
        st.warning("Enter an industry")
        st.stop()

    st.session_state.industry = industry.strip()
    st.session_state.stage = "retrieving"

# ==========================================================
# STEP 2 — RETRIEVE & RANK
# ==========================================================
if st.session_state.stage == "retrieving":
    with st.spinner("Retrieving and ranking pages..."):
        docs = retrieve_docs(st.session_state.industry)
        ranked = rank_documents(st.session_state.industry, docs)

    st.session_state.ranked_docs = ranked

    if assess_confidence(ranked):
        st.session_state.stage = "ready"
    else:
        st.session_state.stage = "need_clarification"

    st.rerun()

# ==========================================================
# STEP 2.1 — ASK CLARIFICATION
# ==========================================================
if st.session_state.stage == "need_clarification":
    st.subheader("Step 2: Clarification Needed")
    st.warning("I am not confident these pages match your intent.")

    clarification = st.text_input("Add more context (country, segment, B2B/B2C)")

    if st.button("Retry with Clarification"):
        st.session_state.clarification_text = clarification
        query = f"{st.session_state.industry} {clarification}"

        with st.spinner("Re-checking..."):
            docs = retrieve_docs(query)
            ranked = rank_documents(query, docs)

        st.session_state.ranked_docs = ranked

        if assess_confidence(ranked):
            st.session_state.stage = "ready"
        else:
            st.session_state.stage = "force_keywords"

        st.rerun()

# ==========================================================
# STEP 2.3 — FORCE KEYWORDS (CLICK = RUN)
# ==========================================================
if st.session_state.stage == "force_keywords":
    st.subheader("Step 2: Choose More Specific Direction")

    suggestions = suggest_force_keywords(
        st.session_state.industry,
        st.session_state.clarification_text
    )

    def run_forced_search(keyword):
        query = f"{st.session_state.industry} {keyword}"
        docs = retrieve_docs(query)
        ranked = rank_documents(query, docs)

        st.session_state.force_keyword_selected = keyword
        st.session_state.ranked_docs = ranked
        st.session_state.stage = "ready"
        st.rerun()

    if suggestions:
        cols = st.columns(len(suggestions))
        for i, k in enumerate(suggestions):
            if cols[i].button(k):
                run_forced_search(k)

    manual = st.text_input("Or type your own specific keyword")

    if st.button("Run Forced Search"):
        run_forced_search(manual)

# ==========================================================
# STEP 2 — SHOW RANKED URLS
# ==========================================================
if st.session_state.stage == "ready":
    st.subheader("Step 2: Top Retrieved Pages")

    ranked = st.session_state.ranked_docs
    strong = [x for x in ranked if x[0] >= 70]

    if len(strong) >= 5:
        st.success("Found 5 strong matches")
        display = ranked[:5]
    else:
        st.warning(f"Only {len(strong)} highly relevant pages found.")
        display = ranked[:5]

    selected_docs = []

    for score, doc in display:
        title = extract_title(doc)
        url = extract_url(doc)
        st.markdown(f"**{title}**")
        st.markdown(f"Relevance: {score}/100")
        st.markdown(url)

        if len(strong) < 5:
            if st.checkbox(f"Include {title}", key=title):
                selected_docs.append(doc)
        else:
            selected_docs.append(doc)

    if st.button("Generate Report"):
        if len(strong) < 5 and not selected_docs:
            st.warning("Select at least one page")
            st.stop()

        docs_for_report = selected_docs if selected_docs else [d for _, d in display]

        report = generate_report(st.session_state.industry, docs_for_report)
        st.session_state.report = report
        st.session_state.stage = "reported"
        st.rerun()

# ==========================================================
# STEP 3 — REPORT
# ==========================================================
if st.session_state.stage == "reported":
    st.subheader("Step 3: Market Research Report")
    st.markdown(st.session_state.report)

    st.caption(f"Word count: {word_count(st.session_state.report)} / 500")

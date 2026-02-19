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
    "Step 1: Enter industry → "
    "Step 2: System finds 5 most relevant Wikipedia pages → "
    "Step 3: Generate structured report under 500 words"
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
    "confidence": "",
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

TOP_K = 10

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

def rank_documents(industry, docs):
    ranked = []
    for d in docs:
        title = extract_title(d)
        preview = d.page_content[:800]

        prompt = f"""
Score relevance of this Wikipedia page to "{industry}".
Respond with ONLY a number 0–100.

Title: {title}
Content: {preview}
"""
        try:
            score_text = llm.invoke(prompt).content.strip()
            score = int(re.findall(r"\d+", score_text)[0])
        except:
            score = 50

        ranked.append((score, d))

    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked

def word_count(text):
    return len(re.findall(r"\b\w+\b", text))

def enforce_under_500(text):
    words = text.split()
    return " ".join(words[:500])

# ==========================================================
# STEP 1 – USER INPUT
# ==========================================================
st.divider()
st.subheader("Step 1: Enter Industry")

industry_input = st.text_input(
    "Industry",
    placeholder="Example: Exotic Pet Care Services"
)

if st.button("Find Relevant Pages", type="primary"):

    if not industry_input.strip():
        st.warning("Please enter an industry.")
        st.stop()

    st.session_state.industry = industry_input.strip()
    st.session_state.clarification_round = 0

    with st.spinner("Retrieving and ranking pages..."):
        docs = retrieve_docs(industry_input)
        ranked = rank_documents(industry_input, docs)

    st.session_state.ranked_docs = ranked

    top_scores = [r[0] for r in ranked[:5]]

    if len(top_scores) >= 5 and all(s >= 70 for s in top_scores):
        st.session_state.docs = [r[1] for r in ranked[:5]]
        st.session_state.stage = "ready"
        st.session_state.confidence = "high"
    else:
        st.session_state.stage = "clarification"
        st.session_state.clarification_round = 1

    st.rerun()

# ==========================================================
# STEP 2 – CLARIFICATION ROUND 1
# ==========================================================
if st.session_state.stage == "clarification" and st.session_state.clarification_round == 1:

    st.subheader("Step 2: Clarification Needed")
    st.warning("System is not fully confident about the top 5 URLs.")

    clarification = st.text_input("Add clarification (country, niche, B2B/B2C etc.)")

    if st.button("Refine Search"):

        refined_query = f"{st.session_state.industry} {clarification}"

        with st.spinner("Re-ranking..."):
            docs = retrieve_docs(refined_query)
            ranked = rank_documents(refined_query, docs)

        st.session_state.ranked_docs = ranked

        top_scores = [r[0] for r in ranked[:5]]

        if len(top_scores) >= 5 and all(s >= 70 for s in top_scores):
            st.session_state.docs = [r[1] for r in ranked[:5]]
            st.session_state.stage = "ready"
        else:
            st.session_state.stage = "force_keywords"
            st.session_state.clarification_round = 2

        st.rerun()

# ==========================================================
# STEP 2 – FORCED KEYWORDS ROUND 2
# ==========================================================
if st.session_state.stage == "force_keywords":

    st.subheader("Step 2: Select Specific Direction")
    st.warning("Still uncertain. Choose a more specific focus.")

    keyword_prompt = f"""
Generate 5 specific sub-keywords related to: {st.session_state.industry}
Return comma-separated only.
"""
    keywords_text = llm.invoke(keyword_prompt).content
    keywords = [k.strip() for k in keywords_text.split(",")]

    selected = None
    cols = st.columns(len(keywords))

    for i, k in enumerate(keywords):
        if cols[i].button(k):
            selected = k

    manual = st.text_input("Or type your own keyword")

    final = selected if selected else manual

    if final:
        final_query = f"{st.session_state.industry} {final}"

        with st.spinner("Final ranking..."):
            docs = retrieve_docs(final_query)
            ranked = rank_documents(final_query, docs)

        st.session_state.ranked_docs = ranked
        st.session_state.docs = [r[1] for r in ranked[:5]]
        st.session_state.stage = "ready"
        st.rerun()

# ==========================================================
# STEP 2 – SHOW RANKED URLS
# ==========================================================
if st.session_state.stage == "ready":

    st.subheader("Step 2: Top Retrieved Pages")

    ranked = st.session_state.ranked_docs
    strong = [r for r in ranked if r[0] >= 70]

    if len(strong) < 5:
        st.info(
            f"{len(strong)} highly relevant pages found (score ≥70). "
            "Other pages shown are best available matches."
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
# STEP 3 – GENERATE REPORT
# ==========================================================
if st.session_state.stage == "ready" and st.session_state.docs:

    st.subheader("Step 3: Generate Report")

    if st.button("Generate Report", type="primary"):

        context = "\n\n".join([d.page_content for d in st.session_state.docs])

        prompt = f"""
You are a professional market research analyst.

STRICT RULES:
- Use ONLY the Wikipedia context provided
- Do NOT invent facts
- Under 500 words
- Clean Markdown structure

Format exactly:

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
"""

        with st.spinner("Generating report..."):
            report = llm.invoke(prompt).content.strip()

        report = enforce_under_500(report)

        st.markdown("## Industry Report")
        st.markdown(report)
        st.caption(f"Word count: {word_count(report)} / 500")

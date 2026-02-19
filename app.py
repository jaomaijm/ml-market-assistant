import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Market Research Assistant", page_icon="📊", layout="wide")
st.title("📊 Market Research Assistant")
st.caption("Retrieve relevant Wikipedia pages and generate a structured industry brief under 500 words.")

# -----------------------------
# Session state
# -----------------------------
defaults = {
    "stage": "start",
    "industry": "",
    "docs": None,
    "final_docs": None,
    "report": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Sidebar: API + LLM
# -----------------------------
st.sidebar.header("Configuration")

with st.sidebar.form("api_form"):
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    llm_option = st.selectbox("Select LLM", ["gpt-4.1-mini"])
    submitted = st.form_submit_button("Use Key")

if submitted:
    st.session_state["api_key"] = api_key
    st.session_state["llm_option"] = llm_option

if "api_key" not in st.session_state or not st.session_state["api_key"]:
    st.info("Enter API key to begin.")
    st.stop()

llm = ChatOpenAI(
    model=st.session_state["llm_option"],
    openai_api_key=st.session_state["api_key"],
    temperature=0.2
)

TOP_K = 8  # retrieve more than 5 for ranking

# -----------------------------
# Helpers
# -----------------------------
def extract_url(doc):
    return doc.metadata.get("source", "")

def extract_title(doc):
    return doc.metadata.get("title", "Wikipedia Page")

def word_count(text):
    return len(re.findall(r"\b\w+\b", text))

def retrieve_docs(query):
    retriever = WikipediaRetriever(top_k_results=TOP_K, lang="en")
    return retriever.invoke(query)

def rank_documents(industry, docs):
    scored = []

    for d in docs:
        prompt = f"""
Industry:
{industry}

Page title:
{extract_title(d)}

Snippet:
{d.page_content[:1000]}

Score relevance 0-10.
Only output number.
"""
        try:
            score = float(re.findall(r"\d+\.?\d*", llm.invoke(prompt).content)[0])
        except:
            score = 0
        scored.append((d, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    strong = [d for d, s in scored if s >= 7]
    weak = [d for d, s in scored if s < 7]

    return strong, weak

def enforce_under_500(text):
    if word_count(text) <= 500:
        return text
    tokens = text.split()
    return " ".join(tokens[:500])

# -----------------------------
# STEP 1: Industry Input
# -----------------------------
st.subheader("Step 1: Enter Industry")

industry = st.text_input("Industry name", placeholder="e.g. Electric Vehicles")

if st.button("Search Wikipedia", type="primary"):
    if not industry.strip():
        st.warning("Please enter an industry.")
        st.stop()

    st.session_state["industry"] = industry
    with st.spinner("Retrieving pages..."):
        docs = retrieve_docs(industry)

    if not docs:
        st.error("No pages found.")
        st.stop()

    st.session_state["docs"] = docs
    st.session_state["stage"] = "ranking"

# -----------------------------
# STEP 2: Ranking + Selection
# -----------------------------
if st.session_state["stage"] == "ranking" and st.session_state["docs"]:

    st.subheader("Step 2: Relevance Ranking")

    strong, weak = rank_documents(
        st.session_state["industry"],
        st.session_state["docs"]
    )

    if len(strong) >= 5:
        st.success("5 highly relevant pages found.")
        st.session_state["final_docs"] = strong[:5]

        for i, d in enumerate(strong[:5], 1):
            st.markdown(f"**{i}. {extract_title(d)}**  \n{extract_url(d)}")

        st.session_state["stage"] = "ready"

    else:
        st.warning(f"Only {len(strong)} strongly relevant pages found.")

        st.markdown("### Strongly Relevant Pages")
        for d in strong:
            st.markdown(f"- {extract_title(d)}  \n{extract_url(d)}")

        needed = 5 - len(strong)
        fallback = weak[:needed]

        st.markdown("### Additional Lower-Relevance Pages")
        for d in fallback:
            st.markdown(f"- {extract_title(d)}  \n{extract_url(d)}")

        combined = strong + fallback
        titles = [extract_title(d) for d in combined]

        selected = st.multiselect(
            "Select up to 5 pages for report:",
            titles,
            default=titles[:len(strong)]
        )

        selected_docs = [d for d in combined if extract_title(d) in selected]

        if len(selected_docs) > 0:
            st.session_state["final_docs"] = selected_docs
            if st.button("Confirm Selection"):
                st.session_state["stage"] = "ready"

# -----------------------------
# STEP 3: Generate Report
# -----------------------------
if st.session_state["stage"] == "ready":

    st.subheader("Step 3: Generate Report")

    if st.button("Generate Industry Report", type="primary"):

        docs = st.session_state["final_docs"]
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are a professional market analyst.

Write a structured industry brief under 500 words.

Industry:
{st.session_state["industry"]}

Use ONLY the context below.

Format:

## Industry Brief

### Scope
(1-2 sentences)

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
(1-2 sentences)

Context:
{context}
"""

        with st.spinner("Generating report..."):
            output = llm.invoke(prompt).content

        report = enforce_under_500(output)
        st.session_state["report"] = report
        st.session_state["stage"] = "done"

# -----------------------------
# OUTPUT
# -----------------------------
if st.session_state["stage"] == "done":
    st.subheader("Industry Report")
    st.markdown(st.session_state["report"])
    st.caption(f"Word count: {word_count(st.session_state['report'])}/500")

    st.download_button(
        "Download Report (TXT)",
        data=st.session_state["report"],
        file_name="industry_report.txt",
        mime="text/plain"
    )

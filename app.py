import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Market Research Assistant", page_icon="📊", layout="wide")
st.title("📊 Market Research Assistant")
st.caption("Structured industry research assistant with clarification + relevance ranking.")

# -----------------------------
# Session State
# -----------------------------
defaults = {
    "stage": "start",
    "industry": "",
    "clarification": "",
    "clarification_round": 0,
    "docs": None,
    "final_docs": None,
    "report": ""
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Configuration")

with st.sidebar.form("api_form"):
    api_key = st.text_input("OpenAI API Key", type="password")
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

TOP_K = 8

# -----------------------------
# Helpers
# -----------------------------
def extract_title(doc):
    return doc.metadata.get("title", "Wikipedia Page")

def extract_url(doc):
    return doc.metadata.get("source", "")

def word_count(text):
    return len(re.findall(r"\b\w+\b", text))

def enforce_under_500(text):
    if word_count(text) <= 500:
        return text
    return " ".join(text.split()[:500])

def retrieve_docs(query):
    retriever = WikipediaRetriever(top_k_results=TOP_K, lang="en")
    return retriever.invoke(query)

def check_ambiguity(industry):
    prompt = f"""
User industry input:
{industry}

If this industry is too broad or ambiguous, ask 2-3 short clarifying questions.
If clear enough for retrieval, output exactly: CLEAR
"""
    result = llm.invoke(prompt).content.strip()
    return result

def rank_documents(industry, docs):
    scored = []

    for d in docs:
        prompt = f"""
Industry:
{industry}

Page title:
{extract_title(d)}

Snippet:
{d.page_content[:800]}

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

# -----------------------------
# STEP 1: Industry Input
# -----------------------------
st.subheader("Step 1: Enter Industry")

industry_input = st.text_input("Industry")

if st.button("Continue", type="primary"):
    if not industry_input.strip():
        st.warning("Please enter industry.")
        st.stop()

    st.session_state["industry"] = industry_input
    st.session_state["clarification_round"] = 1

    result = check_ambiguity(industry_input)

    if result == "CLEAR":
        st.session_state["stage"] = "retrieve"
    else:
        st.session_state["clarification_questions"] = result
        st.session_state["stage"] = "clarify"

# -----------------------------
# STEP 2: Clarification Flow
# -----------------------------
if st.session_state["stage"] == "clarify":

    st.subheader("Clarification Needed")

    st.info(st.session_state["clarification_questions"])

    clarification = st.text_input("Provide more context (country, segment, B2B/B2C etc.)")

    if st.button("Submit Clarification"):

        combined_input = st.session_state["industry"] + " " + clarification
        result = check_ambiguity(combined_input)

        if result == "CLEAR":
            st.session_state["industry"] = combined_input
            st.session_state["stage"] = "retrieve"

        else:
            # Second round escalation
            if st.session_state["clarification_round"] == 1:
                st.session_state["clarification_round"] = 2
                st.session_state["clarification_questions"] = (
                    "Please specify keywords such as:\n"
                    "- Country\n"
                    "- Customer type\n"
                    "- Specific product category\n"
                    "- Market scope (global / regional)"
                )
            else:
                st.session_state["industry"] = combined_input
                st.session_state["stage"] = "retrieve"

# -----------------------------
# STEP 3: Retrieve + Rank
# -----------------------------
if st.session_state["stage"] == "retrieve":

    st.subheader("Step 2: Retrieving Wikipedia Pages")

    with st.spinner("Searching..."):
        docs = retrieve_docs(st.session_state["industry"])

    if not docs:
        st.error("No pages found.")
        st.stop()

    strong, weak = rank_documents(st.session_state["industry"], docs)

    if len(strong) >= 5:
        st.success("5 highly relevant pages found.")
        st.session_state["final_docs"] = strong[:5]

        for i, d in enumerate(strong[:5], 1):
            st.markdown(f"**{i}. {extract_title(d)}**  \n{extract_url(d)}")

        st.session_state["stage"] = "ready"

    else:
        st.warning(f"Only {len(strong)} strongly relevant pages found.")

        fallback = weak[:5 - len(strong)]
        combined = strong + fallback

        titles = [extract_title(d) for d in combined]

        selected = st.multiselect(
            "Select pages for report:",
            titles,
            default=titles
        )

        selected_docs = [d for d in combined if extract_title(d) in selected]

        if selected_docs and st.button("Confirm Selection"):
            st.session_state["final_docs"] = selected_docs
            st.session_state["stage"] = "ready"

# -----------------------------
# STEP 4: Generate Report
# -----------------------------
if st.session_state["stage"] == "ready":

    st.subheader("Step 3: Generate Report")

    if st.button("Generate Report", type="primary"):

        context = "\n\n".join([d.page_content for d in st.session_state["final_docs"]])

        prompt = f"""
Write a structured industry brief under 500 words.

Industry:
{st.session_state["industry"]}

Use ONLY the context below.

Format:

## Industry Brief

### Scope
### Market Offering
### Value Chain
### Competitive Landscape
### Key Trends
### Limits

Context:
{context}
"""

        with st.spinner("Generating..."):
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
        "Download Report",
        data=st.session_state["report"],
        file_name="industry_report.txt",
        mime="text/plain"
    )

import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Market Research Assistant", page_icon="🔎", layout="wide")
st.title("Market Research Assistant")
st.caption(
    "Enter an industry. I will retrieve five relevant Wikipedia pages and produce a report under 500 words based only on those pages. "
    "If I cannot confidently identify the right pages, I will ask for more details instead of guessing."
)

# -----------------------------
# Session state defaults
# -----------------------------
for k, v in {
    "stage": "start",          # start -> need_details -> ready -> reported
    "industry": "",
    "details": "",
    "docs": None,
    "report": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Sidebar: API key + link + reset
# -----------------------------
st.sidebar.header("API Key")
api_key = st.sidebar.text_input("OpenAI API key", type="password", placeholder="sk-...")
st.sidebar.markdown("[Get an API key](https://platform.openai.com/api-keys)")

if st.sidebar.button("Reset", use_container_width=True):
    for k in ["stage", "industry", "details", "docs", "report"]:
        st.session_state[k] = "start" if k == "stage" else "" if k in ["industry", "details", "report"] else None
    st.rerun()

if not api_key:
    st.info("Paste your OpenAI API key in the sidebar to begin")
    st.stop()

# Cheap model for runtime (aligns with assignment hint)
MODEL_NAME = "gpt-4.1-mini"
llm = ChatOpenAI(model=MODEL_NAME, openai_api_key=api_key, temperature=0.2)

TOP_K = 5
LANG = "en"

# -----------------------------
# Helpers
# -----------------------------
def looks_like_input(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 3:
        return False
    if re.fullmatch(r"[\W_]+", t):
        return False
    return True

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))

def extract_url(doc) -> str:
    md = getattr(doc, "metadata", {}) or {}
    return md.get("source") or md.get("url") or ""

def extract_title(doc) -> str:
    md = getattr(doc, "metadata", {}) or {}
    return md.get("title") or md.get("page_title") or "Wikipedia page"

def retrieve_wikipedia_docs(query: str):
    retriever = WikipediaRetriever(top_k_results=TOP_K, lang=LANG)
    return retriever.invoke(query)

def enforce_under_500_words(text: str) -> str:
    if word_count(text) <= 500:
        return text
    compress_prompt = (
        "Shorten the report to UNDER 500 words.\n"
        "Keep headings and key points.\n\n"
        f"{text}"
    )
    return llm.invoke(compress_prompt).content.strip()

def assess_relevance(industry: str, docs) -> dict:
    """
    Use the LLM to judge whether retrieved pages match the user's industry.
    If not confident, ask for more details (country/segment).
    Returns:
      {"status": "ok"} OR {"status": "need_details", "questions": "..."}
    """
    titles = [extract_title(d) for d in docs[:TOP_K]]
    urls = [extract_url(d) for d in docs[:TOP_K]]

    # Keep the context tiny: we only need the titles/urls to judge relevance
    judge_prompt = f"""
You are validating whether Wikipedia pages match the user's intended industry topic.

User industry request:
{industry}

Retrieved pages (title — url):
{chr(10).join([f"- {t} — {u}" for t, u in zip(titles, urls)])}

Task:
1) Decide if these pages are clearly relevant to the user's request.
2) If NOT clearly relevant or the request is ambiguous, do NOT guess. Return ONLY clarifying questions.
3) If clearly relevant, return exactly: OK

Output rules:
- If relevant: output exactly OK
- If not relevant/ambiguous: output 2–4 short clarifying questions only
""".strip()

    verdict = llm.invoke(judge_prompt).content.strip()

    if verdict.strip() == "OK":
        return {"status": "ok"}
    return {"status": "need_details", "questions": verdict}

def build_query(industry: str, details: str) -> str:
    # Details are only used when needed. Otherwise, keep query simple.
    d = (details or "").strip()
    if not d:
        return industry.strip()
    return f"{industry.strip()} ({d})"

# -----------------------------
# Main UI: Step 1
# -----------------------------
st.divider()
st.subheader("Step 1: Tell me the industry")

industry_input = st.text_input(
    "Industry",
    value=st.session_state["industry"],
    placeholder="Example: cosmetics market in Vietnam, online language learning, EV charging",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 1], vertical_alignment="center")
with col1:
    search_clicked = st.button("Find Wikipedia pages", type="primary", use_container_width=True)
with col2:
    st.caption("If your request is too broad, I will ask for details before generating a report")

if search_clicked:
    if not looks_like_input(industry_input):
        st.warning("Please enter an industry to continue")
        st.stop()

    st.session_state["industry"] = industry_input.strip()
    st.session_state["report"] = ""
    st.session_state["docs"] = None
    st.session_state["details"] = ""
    st.session_state["stage"] = "start"

    # -----------------------------
    # Step 2: Retrieve pages
    # -----------------------------
    with st.spinner("Retrieving Wikipedia pages..."):
        docs = retrieve_wikipedia_docs(build_query(st.session_state["industry"], ""))

    if not docs:
        st.session_state["stage"] = "need_details"
        st.rerun()

    # Validate relevance (prevents wrong-topic hallucinations)
    with st.spinner("Checking relevance..."):
        check = assess_relevance(st.session_state["industry"], docs)

    if check["status"] == "need_details":
        st.session_state["stage"] = "need_details"
        st.session_state["docs"] = docs  # keep what we found for transparency
        st.session_state["need_questions"] = check["questions"]
        st.rerun()

    st.session_state["docs"] = docs
    st.session_state["stage"] = "ready"
    st.rerun()

# -----------------------------
# If we need more details, ask user (no optional hint shown by default)
# -----------------------------
if st.session_state["stage"] == "need_details":
    st.subheader("I need a bit more detail to avoid pulling the wrong pages")

    if "need_questions" in st.session_state and st.session_state["need_questions"]:
        st.info(st.session_state["need_questions"])
    else:
        st.info(
            "Please add a short clarification so I can retrieve the right Wikipedia pages (e.g., country, sub-segment, B2B/B2C)."
        )

    details = st.text_input(
        "Clarification",
        value=st.session_state["details"],
        placeholder="Example: Vietnam, consumer beauty products, skincare and cosmetics, B2C",
    )

    colA, colB = st.columns([1, 1], vertical_alignment="center")
    with colA:
        retry = st.button("Retry Wikipedia search", type="primary", use_container_width=True)
    with colB:
        use_anyway = st.button("Use current pages anyway", use_container_width=True)

    if retry:
        if not details.strip():
            st.warning("Please add a short clarification first")
            st.stop()

        st.session_state["details"] = details.strip()

        with st.spinner("Retrieving Wikipedia pages..."):
            docs = retrieve_wikipedia_docs(build_query(st.session_state["industry"], st.session_state["details"]))

        if not docs:
            st.error("Still couldn't retrieve relevant pages. Please try a different wording.")
            st.stop()

        with st.spinner("Checking relevance..."):
            check = assess_relevance(f"{st.session_state['industry']} ({st.session_state['details']})", docs)

        if check["status"] == "need_details":
            st.session_state["docs"] = docs
            st.session_state["need_questions"] = check["questions"]
            st.rerun()

        st.session_state["docs"] = docs
        st.session_state["stage"] = "ready"
        st.rerun()

    if use_anyway:
        if not st.session_state["docs"]:
            st.warning("No pages available to use yet. Please retry the search.")
            st.stop()
        st.session_state["stage"] = "ready"
        st.rerun()

# -----------------------------
# Step 2 display: 5 URLs (title + full link visible)
# -----------------------------
if st.session_state["stage"] in ["ready", "reported"] and st.session_state["docs"]:
    st.subheader("Step 2: Wikipedia pages (5)")

    for i, d in enumerate(st.session_state["docs"][:TOP_K], start=1):
        title = extract_title(d)
        url = extract_url(d) or ""
        # Show title + full URL text, but URL is clickable
        st.markdown(f"**{i}. {title}**  \n{url}  \n[Open page]({url})" if url else f"**{i}. {title}**")

# -----------------------------
# Step 3: Generate report (stable button + spinner)
# -----------------------------
if st.session_state["stage"] in ["ready", "reported"] and st.session_state["docs"]:
    st.subheader("Step 3: Industry report")

    generate = st.button("Generate report", type="primary", use_container_width=True, disabled=bool(st.session_state["report"]))

    if generate:
        docs = st.session_state["docs"]
        context = "\n\n".join([d.page_content for d in docs[:TOP_K]]).strip()

        prompt = f"""
You are a market research assistant for a business analyst at a large corporation.

Write an industry report about: {st.session_state["industry"]}

Rules:
- Use ONLY the Wikipedia context below
- Do NOT invent facts, numbers, competitors, trends, or claims not supported by the context
- If the context is insufficient or ambiguous, output ONLY a short list of clarifying questions (no report)
- Keep the report UNDER 500 words
- Use clear headings and bullet points where helpful
- End with "Limits of this report"

Wikipedia context:
{context}
""".strip()

        with st.spinner("Generating report..."):
            output = llm.invoke(prompt).content.strip()

        # If questions, route back to clarification
        if output.count("?") >= 2 and word_count(output) < 220:
            st.session_state["stage"] = "need_details"
            st.session_state["need_questions"] = output
            st.session_state["report"] = ""
            st.rerun()

        report = enforce_under_500_words(output)
        st.session_state["report"] = report
        st.session_state["stage"] = "reported"
        st.rerun()

# -----------------------------
# Show report + stable download
# -----------------------------
if st.session_state["report"]:
    st.markdown("### Report")
    st.write(st.session_state["report"])

    st.download_button(
        "Download report (TXT)",
        data=st.session_state["report"],
        file_name="industry_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

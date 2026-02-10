import re
import time
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
defaults = {
    "stage": "start",   # start -> need_details -> ready -> reported
    "industry": "",
    "details": "",
    "docs": None,
    "report": "",
    "need_questions": "",
    "download_ready": False,
    "download_preparing": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Sidebar: API key + link + reset
# -----------------------------
st.sidebar.header("API Key")

with st.sidebar.form("api_form", clear_on_submit=False):
    api_key = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
    api_submitted = st.form_submit_button("Use this key", use_container_width=True)

st.sidebar.caption("Your API key is used only for this session to run the app and is not stored")

st.sidebar.markdown("[Get an API key](https://platform.openai.com/api-keys)")

if st.sidebar.button("Reset", use_container_width=True):
    for k in defaults.keys():
        st.session_state[k] = defaults[k]
    st.rerun()

# Persist key for session only (do not store it anywhere else)
if api_submitted:
    st.session_state["api_key_session"] = api_key

api_key_session = st.session_state.get("api_key_session", "")

if not api_key_session:
    st.info("Enter your OpenAI API key in the sidebar and click “Use this key” to begin")
    st.stop()

# Cheap model for runtime
MODEL_NAME = "gpt-4.1-mini"
llm = ChatOpenAI(model=MODEL_NAME, openai_api_key=api_key_session, temperature=0.2)

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
    titles = [extract_title(d) for d in docs[:TOP_K]]
    urls = [extract_url(d) for d in docs[:TOP_K]]

    judge_prompt = f"""
You are validating whether Wikipedia pages match the user's intended industry topic.

User request:
{industry}

Retrieved pages (title — url):
{chr(10).join([f"- {t} — {u}" for t, u in zip(titles, urls)])}

Task:
- If the pages are clearly relevant, output exactly: OK
- If not clearly relevant or the request is ambiguous, output 2–4 short clarifying questions only
- Do not guess or invent

Output rules:
- If relevant: output exactly OK
- Otherwise: output questions only
""".strip()

    verdict = llm.invoke(judge_prompt).content.strip()

    if verdict.strip() == "OK":
        return {"status": "ok"}
    return {"status": "need_details", "questions": verdict}

def build_query(industry: str, details: str) -> str:
    d = (details or "").strip()
    if not d:
        return industry.strip()
    return f"{industry.strip()} ({d})"

def should_ask_for_more_context(docs) -> bool:
    if not docs or len(docs) < 3:
        return True
    joined = "\n\n".join([getattr(d, "page_content", "") for d in docs if getattr(d, "page_content", "")])
    return len(joined.strip()) < 1200

# -----------------------------
# Main UI
# -----------------------------
st.divider()
st.subheader("Step 1: Tell me the industry")

industry_input = st.text_input(
    "Industry",
    value=st.session_state["industry"],
    placeholder="Example: cosmetics market in Vietnam",
    label_visibility="collapsed",
)

# Equal-width buttons
b1, b2 = st.columns(2)
with b1:
    search_clicked = st.button("Find Wikipedia pages", type="primary", use_container_width=True)
with b2:
    clear_clicked = st.button("Clear results", use_container_width=True)

if clear_clicked:
    for k in ["stage", "details", "docs", "report", "need_questions", "download_ready", "download_preparing"]:
        st.session_state[k] = defaults[k]
    st.session_state["industry"] = industry_input.strip()
    st.rerun()

if search_clicked:
    if not looks_like_input(industry_input):
        st.warning("Please enter an industry to continue")
        st.stop()

    st.session_state["industry"] = industry_input.strip()
    st.session_state["details"] = ""
    st.session_state["docs"] = None
    st.session_state["report"] = ""
    st.session_state["need_questions"] = ""
    st.session_state["download_ready"] = False
    st.session_state["download_preparing"] = False
    st.session_state["stage"] = "start"

    with st.spinner("Retrieving Wikipedia pages..."):
        docs = retrieve_wikipedia_docs(build_query(st.session_state["industry"], ""))

    if not docs:
        st.session_state["stage"] = "need_details"
        st.rerun()

    with st.spinner("Checking relevance..."):
        check = assess_relevance(st.session_state["industry"], docs)

    if check["status"] == "need_details":
        st.session_state["stage"] = "need_details"
        st.session_state["docs"] = docs
        st.session_state["need_questions"] = check["questions"]
        st.rerun()

    st.session_state["docs"] = docs
    st.session_state["stage"] = "ready"
    st.rerun()

# -----------------------------
# Need details flow
# -----------------------------
if st.session_state["stage"] == "need_details":
    st.subheader("I need a bit more detail to avoid pulling the wrong pages")

    if st.session_state["need_questions"]:
        st.info(st.session_state["need_questions"])
    else:
        st.info("Please add a short clarification (country, sub-segment, B2B/B2C)")

    details = st.text_input(
        "Clarification",
        value=st.session_state["details"],
        placeholder="Example: Vietnam, consumer cosmetics, skincare and makeup, B2C",
    )

    c1, c2 = st.columns(2)
    with c1:
        retry = st.button("Retry Wikipedia search", type="primary", use_container_width=True)
    with c2:
        use_anyway = st.button("Use current pages", use_container_width=True)

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
# Step 2 display (title + full URL, no extra "Open page")
# -----------------------------
if st.session_state["stage"] in ["ready", "reported"] and st.session_state["docs"]:
    st.subheader("Step 2: Wikipedia pages (5)")

    for i, d in enumerate(st.session_state["docs"][:TOP_K], start=1):
        title = extract_title(d)
        url = extract_url(d) or ""
        if url:
            st.markdown(f"**{i}. {title}**  \n{url}")
        else:
            st.markdown(f"**{i}. {title}**")

# -----------------------------
# Step 3 generate report
# -----------------------------
if st.session_state["stage"] in ["ready", "reported"] and st.session_state["docs"]:
    st.subheader("Step 3: Industry report")

    g1, g2 = st.columns(2)
    with g1:
        generate = st.button(
            "Generate report",
            type="primary",
            use_container_width=True,
            disabled=bool(st.session_state["report"]),
        )
    with g2:
        regenerate = st.button(
            "Regenerate",
            use_container_width=True,
            disabled=not bool(st.session_state["report"]),
        )

    if regenerate:
        st.session_state["report"] = ""
        st.session_state["download_ready"] = False
        st.session_state["download_preparing"] = False
        st.session_state["stage"] = "ready"
        st.rerun()

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

        if output.count("?") >= 2 and word_count(output) < 220:
            st.session_state["stage"] = "need_details"
            st.session_state["need_questions"] = output
            st.session_state["report"] = ""
            st.session_state["download_ready"] = False
            st.session_state["download_preparing"] = False
            st.rerun()

        report = enforce_under_500_words(output)
        st.session_state["report"] = report
        st.session_state["stage"] = "reported"
        st.session_state["download_ready"] = True
        st.session_state["download_preparing"] = False
        st.rerun()

# -----------------------------
# Report + download status
# -----------------------------
if st.session_state["report"]:
    st.markdown("### Report")
    st.write(st.session_state["report"])
      # Word count
    wc = word_count(st.session_state["report"])
    st.caption(f"Word count: {wc} / 500")

    # Download status area
   if st.session_state["report"]:
    st.markdown("### Report")
    st.write(st.session_state["report"])

    wc = word_count(st.session_state["report"])
    st.caption(f"Word count: {wc} / 500")

    st.download_button(
        "Download report (TXT)",
        data=st.session_state["report"],
        file_name="industry_report.txt",
        mime="text/plain",
        use_container_width=True,
    )


    d1, d2 = st.columns(2)
    with d1:
        # Simulated "preparing" status on click (Streamlit doesn't expose actual file-transfer progress)
        prep = st.button("Prepare download", use_container_width=True)
    with d2:
        download_disabled = not st.session_state["download_ready"] or st.session_state["download_preparing"]
        download_clicked = st.download_button(
            "Download report (TXT)",
            data=st.session_state["report"],
            file_name="industry_report.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=download_disabled,
        )

    if prep:
        st.session_state["download_preparing"] = True
        st.session_state["download_ready"] = False
        st.rerun()

    if st.session_state["download_preparing"]:
        # Small delay to make status visible and feel responsive
        time.sleep(0.6)
        st.session_state["download_preparing"] = False
        st.session_state["download_ready"] = True
        st.rerun()

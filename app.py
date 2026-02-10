import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Market Research Assistant", page_icon="🔎", layout="wide")
st.title("🔎 Market Research Assistant")
st.caption("Tell me an industry → I’ll pull 5 relevant Wikipedia pages → then write a <500-word report based on those pages only")

# -----------------------------
# Session state defaults
# -----------------------------
for k, v in {
    "did_search": False,
    "industry": "",
    "clarification": "",
    "docs": None,
    "query_used": "",
    "report": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Sidebar: API key + link + reset
# -----------------------------
st.sidebar.header("API Key")
api_key = st.sidebar.text_input("OpenAI API key", type="password", placeholder="sk-...")
st.sidebar.caption("Used only for this session and not stored")

st.sidebar.markdown("[Get an API key here](https://platform.openai.com/api-keys)")

if st.sidebar.button("🔄 Reset", use_container_width=True):
    for k in ["did_search", "industry", "clarification", "docs", "query_used", "report"]:
        st.session_state[k] = "" if isinstance(st.session_state[k], str) else False if isinstance(st.session_state[k], bool) else None
    st.rerun()

if not api_key:
    st.info("Paste your OpenAI API key in the left sidebar to start")
    st.stop()

# Use a cheap model for build/runtime (aligns with assignment hint)
MODEL_NAME = "gpt-4.1-mini"
llm = ChatOpenAI(model=MODEL_NAME, openai_api_key=api_key, temperature=0.2)

TOP_K = 5
LANG = "en"

# -----------------------------
# Helper functions
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

def retrieve_wikipedia_docs(query: str, top_k: int = TOP_K, lang: str = LANG):
    retriever = WikipediaRetriever(top_k_results=top_k, lang=lang)
    return retriever.invoke(query)

def get_docs_with_fallbacks(industry_text: str):
    candidates = [
        industry_text,
        f"{industry_text} industry",
        f"{industry_text} market",
        f"{industry_text} sector",
    ]
    for q in candidates:
        docs = retrieve_wikipedia_docs(q, TOP_K, LANG)
        if docs and len(docs) > 0:
            return docs, q
    return [], industry_text

def should_ask_for_more_context(docs) -> bool:
    if not docs or len(docs) < 3:
        return True
    joined = "\n\n".join([getattr(d, "page_content", "") for d in docs if getattr(d, "page_content", "")])
    return len(joined.strip()) < 1200

def build_clarifying_questions(industry_term: str) -> str:
    return "\n".join(
        [
            "I couldn’t confidently find the right Wikipedia pages for that yet, so I don’t want to guess",
            "Can you clarify quickly",
            "",
            f"- When you say “{industry_term}”, what type or sub-segment do you mean",
            "- Any geography focus (global, UK, Thailand, etc)",
            "- Is this mainly B2B, B2C, or both",
            "- Any example companies/products in this industry",
        ]
    )

def enforce_under_500_words(text: str) -> str:
    if word_count(text) <= 500:
        return text
    compress_prompt = (
        "Shorten the report to UNDER 500 words.\n"
        "Keep headings and key points.\n\n"
        f"{text}"
    )
    return llm.invoke(compress_prompt).content.strip()

# -----------------------------
# Fun stepper / progress
# -----------------------------
step = 1
if st.session_state["did_search"]:
    step = 2
if st.session_state["docs"]:
    step = 3

st.progress(step / 3)
st.caption(f"Step {step} of 3")

st.divider()

# -----------------------------
# Step 1: Form (stable UI)
# -----------------------------
st.subheader("1) What industry should I research 🧠")

with st.form("search_form", clear_on_submit=False):
    industry = st.text_input(
        "Industry",
        placeholder="Example: online language learning, EV charging, pet food",
        value=st.session_state["industry"],
        label_visibility="collapsed",
    )
    clarification = st.text_input(
        "Optional hint (keeps results accurate)",
        placeholder="Example: UK market, mobile apps, subscription, B2B…",
        value=st.session_state["clarification"],
    )
    submitted = st.form_submit_button("🔍 Find Wikipedia pages", use_container_width=True)

if not looks_like_input(industry):
    st.info("Type an industry above to begin")
    st.stop()

if submitted:
    st.session_state["industry"] = industry.strip()
    st.session_state["clarification"] = clarification.strip()
    st.session_state["did_search"] = True
    st.session_state["docs"] = None
    st.session_state["report"] = ""
    st.rerun()

# -----------------------------
# Step 2: Retrieve pages (auto after submit)
# -----------------------------
if st.session_state["did_search"] and not st.session_state["docs"]:
    st.subheader("2) Finding the best Wikipedia pages 📚")

    query_base = st.session_state["industry"]
    if st.session_state["clarification"]:
        query_base = f"{query_base} ({st.session_state['clarification']})"

    with st.status("🕵️ Searching Wikipedia…", expanded=False) as status:
        docs, query_used = get_docs_with_fallbacks(query_base)
        status.update(label="✅ Wikipedia search completed", state="complete")

    if not docs or should_ask_for_more_context(docs):
        st.warning(build_clarifying_questions(st.session_state["industry"]))
        st.stop()

    st.session_state["docs"] = docs
    st.session_state["query_used"] = query_used
    st.rerun()

# -----------------------------
# Display Step 2 results (cards)
# -----------------------------
if st.session_state["docs"]:
    st.subheader("2) Wikipedia pages I used 📚")
    st.caption("These are the sources for the report (5 pages)")

    for i, d in enumerate(st.session_state["docs"][:TOP_K], start=1):
        title = extract_title(d)
        url = extract_url(d)

        with st.container(border=True):
            st.markdown(f"**{i}. {title}**")
            if url:
                st.markdown(f"[Open page]({url})")
            else:
                st.caption("URL not available for this result, but content was retrieved")

# -----------------------------
# Step 3: Generate report
# -----------------------------
if st.session_state["docs"]:
    st.subheader("3) Your industry report ✍️")
    st.caption("Rules: under 500 words, and only based on the Wikipedia pages above")

    gen = st.button("🧾 Generate my report", type="primary", use_container_width=True)

    if gen:
        docs = st.session_state["docs"]
        context = "\n\n".join([d.page_content for d in docs[:TOP_K]]).strip()

        prompt = f"""
You are a market research assistant for a business analyst at a large corporation.

Write an industry report about: {st.session_state["industry"]}

Rules:
- Use ONLY the Wikipedia context below
- Do NOT invent facts, numbers, competitors, trends, or claims not supported by the context
- If the context is insufficient or ambiguous, output ONLY a short list of clarifying questions (no report)
- If you can write the report, keep it UNDER 500 words
- Use clear headings and bullet points where helpful
- End with "Limits of this report" (limitations of relying on Wikipedia)

Wikipedia context:
{context}
""".strip()

        with st.status("🤖 Writing your report…", expanded=False) as status:
            output = llm.invoke(prompt).content.strip()
            status.update(label="✅ Done", state="complete")

        # If it looks like questions instead of a report, show and stop
        if output.count("?") >= 2 and word_count(output) < 200:
            st.warning(output)
            st.stop()

        report = enforce_under_500_words(output)
        st.session_state["report"] = report
        st.rerun()

# -----------------------------
# Show report if available
# -----------------------------
if st.session_state["report"]:
    st.markdown("### ✅ Industry report")
    st.write(st.session_state["report"])

    st.caption(f"Word count: {word_count(st.session_state['report'])} (must be under 500)")

    st.download_button(
        "⬇️ Download report (TXT)",
        data=st.session_state["report"],
        file_name="industry_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

import re
import streamlit as st

from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Market Research Assistant", page_icon="🔎", layout="wide")
st.title("🔎 Market Research Assistant")
st.caption("Tell me an industry → I’ll pull the 5 most relevant Wikipedia pages → then write a <500-word report based on those pages only")

# -----------------------------
# Sidebar: API key only + link
# -----------------------------
st.sidebar.header("API Key")
api_key = st.sidebar.text_input("OpenAI API key", type="password", placeholder="sk-...")
st.sidebar.caption("Your key is used only for this session and is not stored")

st.sidebar.markdown(
    "Need a key? Get it here: "
    "[OpenAI API Keys](https://platform.openai.com/api-keys)"
)

if not api_key:
    st.info("Paste your OpenAI API key in the left sidebar to start")
    st.stop()

# Fixed settings (per your request)
TOP_K = 5
LANG = "en"
MODEL_NAME = "gpt-4.1-mini"  # cheap + good enough for this assignment

llm = ChatOpenAI(model=MODEL_NAME, openai_api_key=api_key, temperature=0.2)

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

def extract_url(doc) -> str:
    md = getattr(doc, "metadata", {}) or {}
    return md.get("source") or md.get("url") or ""

def extract_title(doc) -> str:
    md = getattr(doc, "metadata", {}) or {}
    return md.get("title") or md.get("page_title") or "Wikipedia page"

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))

def retrieve_wikipedia_docs(query: str, top_k: int = TOP_K, lang: str = LANG):
    retriever = WikipediaRetriever(top_k_results=top_k, lang=lang)
    return retriever.invoke(query)

def get_docs_with_fallbacks(industry_text: str):
    # Try a few safe query variants before asking the user for more context
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

def enforce_under_500_words(report: str) -> str:
    if word_count(report) <= 500:
        return report
    compress_prompt = (
        "Shorten the report to UNDER 500 words.\n"
        "Keep headings and key points.\n\n"
        f"{report}"
    )
    return llm.invoke(compress_prompt).content.strip()

def build_clarifying_questions(industry_term: str) -> str:
    return "\n".join(
        [
            "I couldn’t confidently identify the right Wikipedia pages for that industry yet",
            "Can you clarify a bit so I don’t guess",
            "",
            f"- When you say “{industry_term}”, what type or sub-segment do you mean",
            "- Any geography focus (global, UK, Thailand, etc)",
            "- Is this mainly B2B, B2C, or both",
            "- Any specific examples of companies/products in this industry",
        ]
    )

def should_ask_for_more_context(docs) -> bool:
    # If we didn’t get enough pages/content, don’t force a report
    if not docs or len(docs) < 3:
        return True
    joined = "\n\n".join([d.page_content for d in docs if getattr(d, "page_content", "")])
    # conservative: if context is tiny, it’s risky
    return len(joined.strip()) < 1200

# -----------------------------
# Main UI
# -----------------------------
st.divider()
st.subheader("1) Tell me the industry you want to research")
industry = st.text_input(
    "Industry",
    placeholder="Example: online language learning, EV charging, pet food",
    label_visibility="collapsed",
)

if not looks_like_input(industry):
    st.warning("Type an industry to continue")
    st.stop()

# Session state to allow “ask for more context” flow without clutter
if "clarification" not in st.session_state:
    st.session_state.clarification = ""

st.subheader("2) I’ll find the most relevant Wikipedia pages")
colA, colB = st.columns([1, 2], vertical_alignment="center")

with colA:
    fetch_clicked = st.button("Find Wikipedia pages", type="primary", use_container_width=True)

with colB:
    st.caption("Tip: If your industry term is vague, I may ask you to clarify so the report stays accurate")

docs = None
query_used = None

if fetch_clicked:
    with st.status("Searching Wikipedia…", expanded=False) as status:
        # If we already have clarification, use it to disambiguate
        query_base = industry
        if st.session_state.clarification.strip():
            query_base = f"{industry} ({st.session_state.clarification.strip()})"

        docs, query_used = get_docs_with_fallbacks(query_base)
        status.update(label="Wikipedia search completed", state="complete")

    if not docs or should_ask_for_more_context(docs):
        st.warning(build_clarifying_questions(industry))
        st.session_state.clarification = st.text_area(
            "Add a short clarification (then click “Find Wikipedia pages” again)",
            placeholder="Example: mobile apps, subscription-based, UK market, B2C learners…",
            height=90,
        )
        st.stop()

    # Show 5 results as clean, clickable titles
    st.markdown("Here are the 5 most relevant Wikipedia pages I found")
    shown = 0
    for d in docs:
        url = extract_url(d)
        title = extract_title(d)
        if url:
            shown += 1
            st.markdown(f"- [{title}]({url})")
        if shown >= 5:
            break

    if shown < 5:
        st.info("I retrieved page content successfully, but some results didn’t include a clean URL in metadata")

    # Store docs in session so Step 3 can run without re-fetching
    st.session_state["docs"] = docs
    st.session_state["query_used"] = query_used

# If pages already fetched earlier, display them again (nice UX)
if "docs" in st.session_state and st.session_state["docs"]:
    st.subheader("3) Generate your industry report")
    st.caption("Report rules: <500 words, and based only on the 5 Wikipedia pages above")

    if st.button("Generate report", use_container_width=True):
        docs = st.session_state["docs"]

        context_chunks = [d.page_content for d in docs[:TOP_K]]
        context = "\n\n".join(context_chunks).strip()

        # Hard instruction: if insufficient/ambiguous, ask questions only (no hallucination)
        prompt = f"""
You are a market research assistant for a business analyst at a large corporation

Write an industry report about: {industry}

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

        with st.status("Writing report…", expanded=False) as status:
            output = llm.invoke(prompt).content.strip()
            status.update(label="Done", state="complete")

        # If model asks questions, show them and stop
        if output.count("?") >= 2 and word_count(output) < 180:
            st.warning(output)
            st.session_state.clarification = st.text_area(
                "Add a short clarification (then click “Find Wikipedia pages” again)",
                placeholder="Example: mobile apps, subscription-based, UK market, B2C learners…",
                height=90,
            )
            st.stop()

        report = enforce_under_500_words(output)

        st.markdown("### Industry report")
        st.write(report)

        st.download_button(
            "Download report (TXT)",
            data=report,
            file_name="industry_report.txt",
            mime="text/plain",
            use_container_width=True,
        )


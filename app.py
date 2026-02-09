import re
import streamlit as st

from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Market Research Assistant", page_icon="📚", layout="wide")
st.title("📚 Market Research Assistant")
st.caption("Step 1 validate industry → Step 2 show 5 Wikipedia URLs → Step 3 generate a <500-word report (or ask for more context if unsure)")

# -----------------------------
# Sidebar: API key + settings
# -----------------------------
st.sidebar.header("Settings")

api_key = st.sidebar.text_input("OpenAI API key", type="password", placeholder="sk-...")
st.sidebar.caption("Key is used only for this session and is not stored")

mode = st.sidebar.radio("Model mode", ["Build (cheap)", "Check (more capable)"], horizontal=False)
model_name = "gpt-4.1-mini" if mode == "Build (cheap)" else "gpt-4.1"

st.sidebar.markdown("---")
st.sidebar.subheader("Wikipedia retrieval")
top_k = st.sidebar.slider("Number of Wikipedia pages", min_value=5, max_value=10, value=5, step=1)
lang = st.sidebar.selectbox("Language", ["en"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Uncertainty handling")
min_context_chars = st.sidebar.slider("Min context length before reporting", 1500, 12000, 4500, 500)
st.sidebar.caption("If retrieved Wikipedia context is too thin, the assistant will ask clarifying questions instead of guessing")

if not api_key:
    st.info("Enter your OpenAI API key in the left sidebar to begin")
    st.stop()

# -----------------------------
# Step 1: Input validation
# -----------------------------
st.subheader("Step 1: Provide an industry")
industry = st.text_input("Industry", placeholder="e.g., online language learning, EV charging, pet food")

def looks_like_industry(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 3:
        return False
    if re.fullmatch(r"[\W_]+", t):  # only symbols
        return False
    return True

if not looks_like_industry(industry):
    st.warning("Please enter an industry (at least a few characters) to continue")
    st.stop()

# Optional: user-provided context to reduce ambiguity (keeps it KISS but prevents hallucination)
st.markdown("Optional: add a bit more context if your industry term is broad or ambiguous")
extra_context = st.text_area(
    "Extra context (optional)",
    placeholder="Example: geography (UK/Thailand), B2B vs B2C, which segment (apps, tutors, testing), time horizon, etc",
    height=90
)

st.success("Industry received")

# -----------------------------
# Helpers: retrieval + generation
# -----------------------------
@st.cache_data(show_spinner=False)
def retrieve_wikipedia_docs(query: str, top_k_results: int, language: str):
    retriever = WikipediaRetriever(top_k_results=top_k_results, lang=language)
    # Newer LangChain retrievers use .invoke()
    docs = retriever.invoke(query)
    return docs

def extract_url(doc) -> str:
    md = getattr(doc, "metadata", {}) or {}
    return md.get("source") or md.get("url") or ""

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))

def enforce_under_500_words(llm, report: str) -> str:
    if word_count(report) <= 500:
        return report
    compress_prompt = (
        "Shorten the report to UNDER 500 words.\n"
        "Keep the same headings and preserve key points.\n\n"
        f"{report}"
    )
    return llm.invoke(compress_prompt).content.strip()

def is_context_too_thin(context: str, min_chars: int) -> bool:
    # Conservative heuristic: if we don't have enough retrieved text, we should ask questions instead of guessing
    clean = (context or "").strip()
    return len(clean) < min_chars

# -----------------------------
# Step 2: Retrieve and show URLs
# -----------------------------
st.subheader("Step 2: Top Wikipedia pages (URLs)")

with st.spinner("Retrieving Wikipedia pages..."):
    docs = retrieve_wikipedia_docs(industry, top_k, lang)

if not docs:
    st.error("No Wikipedia results found. Try a broader industry term, or add extra context")
    st.stop()

urls = []
contexts = []

for d in docs[:top_k]:
    url = extract_url(d)
    if url:
        urls.append(url)
    contexts.append(d.page_content)

# Display URLs (aim for 5)
for i, u in enumerate(urls[:5], start=1):
    st.write(f"{i}. {u}")

if len(urls) < 5:
    st.info("Some Wikipedia results did not provide URLs via metadata, but their content was still retrieved")

context = "\n\n".join(contexts[:top_k]).strip()
context_len = len(context)

st.caption(f"Retrieved context length: {context_len} characters")

# -----------------------------
# Step 3: Generate report (<500 words) OR ask for more context if unsure
# -----------------------------
st.subheader("Step 3: Generate output (<500 words)")

report_focus = st.selectbox(
    "Choose report focus",
    [
        "Market overview (default)",
        "Competitive snapshot",
        "Customer segments focus",
        "Value chain focus",
    ],
)

focus_hint = {
    "Market overview (default)": "Explain what the industry is, how it works, typical players, and key trends",
    "Competitive snapshot": "Emphasize categories of competitors, differentiation, and where competition happens",
    "Customer segments focus": "Emphasize who the customers are, use cases, and segment differences",
    "Value chain focus": "Emphasize upstream/downstream activities, distribution, and where value is created",
}[report_focus]

llm = ChatOpenAI(
    model=model_name,
    openai_api_key=api_key,
    temperature=0.2,
)

def build_questions(industry_term: str) -> str:
    # Keep it short and practical for the user to answer in the app
    return "\n".join([
        "I need a bit more context to avoid guessing. Quick questions:",
        f"- When you say **{industry_term}**, which sub-segment do you mean (examples: product type, channel, customer type)?",
        "- Which geography should the report focus on (global, UK, Thailand, etc)?",
        "- Is the audience B2B or B2C (or both)?",
        "- Any specific time horizon (current snapshot vs next 3–5 years)?",
    ])

if st.button("Generate", type="primary"):
    # Hard guardrail: if retrieval is thin, ask for context (no hallucination)
    if is_context_too_thin(context, min_context_chars):
        st.warning("Wikipedia context retrieved is too limited for a reliable report")
        st.markdown(build_questions(industry))
        st.stop()

    # LLM guardrail: force it to ask questions instead of inventing facts
    prompt = f"""
You are a market research assistant for a business analyst at a large corporation.

You must follow these rules:
- Use ONLY the Wikipedia context below
- If the context is insufficient, ambiguous, or does not support a claim, DO NOT guess
- If insufficient, output ONLY a short list of clarifying questions (no report)
- Otherwise, write an industry report UNDER 500 words
- Use clear headings and bullet points where useful
- End with a short section titled "Limits of this report" describing limitations of relying on Wikipedia
- Never invent competitor names, market sizes, growth rates, or trends unless explicitly supported by the context

Topic/industry: {industry}

User-provided extra context (may be empty):
{extra_context}

Report focus:
{focus_hint}

Wikipedia context:
{context}
""".strip()

    with st.spinner(f"Generating with {model_name}..."):
        output = llm.invoke(prompt).content.strip()

    # If the model chose to ask questions, show them as-is
    if output.lower().startswith("i need") or "clarifying question" in output.lower() or output.strip().endswith("?"):
        st.markdown(output)
        st.stop()

    # Otherwise treat as report and enforce word limit
    report = enforce_under_500_words(llm, output)

    st.markdown("#### Output")
    st.write(report)
    st.caption(f"Word count: {word_count(report)} (must be under 500)")

    st.download_button(
        "Download as TXT",
        data=report,
        file_name="industry_report.txt",
        mime="text/plain",
    )

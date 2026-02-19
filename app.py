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
defaults = {
    "stage": "start",   # start -> need_details -> keyword_force -> ready -> reported
    "industry": "",
    "details": "",
    "docs": None,
    "report": "",
    "need_questions": "",
    "download_ready": False,
    "download_preparing": False,
    "need_attempts": 0,               # NEW: count how many times we needed clarification
    "suggested_keywords": [],         # NEW: keywords suggested by LLM
    "forced_keywords": "",            # NEW: the chosen/entered keywords (string)
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Sidebar: API key + (single-option) LLM dropdown + reset
# -----------------------------
st.sidebar.header("API Key & LLM")

with st.sidebar.form("api_form", clear_on_submit=False):
    api_key = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
    llm_option = st.selectbox(
        "Select LLM",
        options=["gpt-4.1-mini"],  # final version: only one option
        index=0
    )
    api_submitted = st.form_submit_button("Use this key", use_container_width=True)

st.sidebar.caption("Your API key is used only for this session to run the app and is not stored")
st.sidebar.markdown("[Get an API key](https://platform.openai.com/api-keys)")

if st.sidebar.button("Reset", use_container_width=True):
    for k in defaults.keys():
        st.session_state[k] = defaults[k]
    st.rerun()

if api_submitted:
    st.session_state["api_key_session"] = api_key
    st.session_state["llm_option"] = llm_option

api_key_session = st.session_state.get("api_key_session", "")
llm_option = st.session_state.get("llm_option", "gpt-4.1-mini")

if not api_key_session:
    st.info("Enter your OpenAI API key in the sidebar and click “Use this key” to begin")
    st.stop()

# LLM
MODEL_NAME = llm_option
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

def build_query(industry: str, details: str) -> str:
    d = (details or "").strip()
    if not d:
        return industry.strip()
    return f"{industry.strip()} ({d})"

def enforce_under_500_words(text: str, limit: int = 500, max_rounds: int = 3) -> str:
    """
    Guarantees final text <= limit words.
    1) Ask the model to compress (up to max_rounds)
    2) If still too long, hard-trim tokens (last resort)
    """
    current = (text or "").strip()

    for _ in range(max_rounds):
        if word_count(current) <= limit:
            return current

        compress_prompt = f"""
Shorten this report to UNDER {limit} words.

Rules:
- Keep headings and bullets
- Remove less important details first
- Do NOT add new facts
- Output ONLY the revised text

Report:
{current}
""".strip()

        current = llm.invoke(compress_prompt).content.strip()

    # last resort: hard trim by whitespace tokens
    tokens = current.split()
    return " ".join(tokens[:limit])

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

def suggest_keywords(industry: str) -> list[str]:
    """
    NEW: on 2nd failure, propose specific keywords to force a Wikipedia retrieval.
    """
    kw_prompt = f"""
You help users refine a Wikipedia search query.

User industry input:
{industry}

Output 8–12 concise keywords/phrases (1–4 words each) that are likely to return relevant Wikipedia pages.
Rules:
- Output ONLY a bullet list
- No explanations
- Keywords should narrow scope (segment, geography, value chain, market terms, regulation, technology, key product categories)
""".strip()

    raw = llm.invoke(kw_prompt).content.strip()
    # parse bullets -> list
    lines = [re.sub(r"^[-•\s]+", "", ln).strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln and len(ln) <= 40]
    # de-dup while preserving order
    seen = set()
    out = []
    for ln in lines:
        key = ln.lower()
        if key not in seen:
            seen.add(key)
            out.append(ln)
    return out[:12]

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

b1, b2 = st.columns(2)
with b1:
    search_clicked = st.button("Find Wikipedia pages", type="primary", use_container_width=True)
with b2:
    clear_clicked = st.button("Clear results", use_container_width=True)

if clear_clicked:
    for k in ["stage", "details", "docs", "report", "need_questions",
              "download_ready", "download_preparing", "need_attempts",
              "suggested_keywords", "forced_keywords"]:
        st.session_state[k] = defaults[k]
    st.session_state["industry"] = industry_input.strip()
    st.rerun()

# -----------------------------
# Search flow
# -----------------------------
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
    st.session_state["need_attempts"] = 0
    st.session_state["suggested_keywords"] = []
    st.session_state["forced_keywords"] = ""
    st.session_state["stage"] = "start"

    with st.spinner("Retrieving Wikipedia pages..."):
        docs = retrieve_wikipedia_docs(build_query(st.session_state["industry"], ""))

    if not docs:
        st.session_state["stage"] = "need_details"
        st.session_state["need_attempts"] = 1
        st.rerun()

    with st.spinner("Checking relevance..."):
        check = assess_relevance(st.session_state["industry"], docs)

    if check["status"] == "need_details":
        st.session_state["stage"] = "need_details"
        st.session_state["docs"] = docs
        st.session_state["need_questions"] = check["questions"]
        st.session_state["need_attempts"] = 1
        st.rerun()

    st.session_state["docs"] = docs
    st.session_state["stage"] = "ready"
    st.rerun()

# -----------------------------
# Need details flow (1st time)
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
            # 2nd failure -> keyword forcing
            st.session_state["need_attempts"] = 2
            st.session_state["stage"] = "keyword_force"
            st.rerun()

        with st.spinner("Checking relevance..."):
            check = assess_relevance(f"{st.session_state['industry']} ({st.session_state['details']})", docs)

        if check["status"] == "need_details":
            # 2nd failure -> keyword forcing
            st.session_state["docs"] = docs
            st.session_state["need_questions"] = check["questions"]
            st.session_state["need_attempts"] = 2
            st.session_state["stage"] = "keyword_force"
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
# NEW: Keyword forcing flow (2nd time)
# -----------------------------
if st.session_state["stage"] == "keyword_force":
    st.subheader("Still unclear — choose keywords to force a precise search")

    # show the latest questions so user understands why
    if st.session_state["need_questions"]:
        st.info(st.session_state["need_questions"])

    # generate keyword suggestions once
    if not st.session_state["suggested_keywords"]:
        with st.spinner("Generating keyword suggestions..."):
            st.session_state["suggested_keywords"] = suggest_keywords(st.session_state["industry"])

    st.caption("Pick a few keywords (or type your own). Then I will force-retrieve 5 Wikipedia pages.")

    picked = st.multiselect(
        "Suggested keywords",
        options=st.session_state["suggested_keywords"],
        default=st.session_state["suggested_keywords"][:4] if st.session_state["suggested_keywords"] else [],
    )

    custom = st.text_input(
        "Add custom keywords (comma-separated)",
        value=st.session_state["forced_keywords"],
        placeholder="Example: Vietnam, pet food, premiumization, supply chain",
    )

    k1, k2 = st.columns(2)
    with k1:
        force_search = st.button("Force Wikipedia pages", type="primary", use_container_width=True)
    with k2:
        back_to_details = st.button("Back to clarification", use_container_width=True)

    if back_to_details:
        st.session_state["stage"] = "need_details"
        st.rerun()

    if force_search:
        custom_list = [x.strip() for x in (custom or "").split(",") if x.strip()]
        all_keywords = [*picked, *custom_list]
        all_keywords = [k for k in all_keywords if k]

        if not all_keywords:
            st.warning("Please select or enter at least 1 keyword")
            st.stop()

        st.session_state["forced_keywords"] = ", ".join(all_keywords)

        forced_query = f"{st.session_state['industry']} " + " ".join(all_keywords)

        with st.spinner("Forcing Wikipedia retrieval..."):
            docs = retrieve_wikipedia_docs(forced_query)

        if not docs:
            st.error("Still couldn't retrieve pages. Try different keywords (more specific terms).")
            st.stop()

        # IMPORTANT: We force-ready (skip relevance loop) to guarantee 5 URLs display
        st.session_state["docs"] = docs
        st.session_state["stage"] = "ready"
        st.rerun()

# -----------------------------
# Step 2 display (title + full URL)
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
You are a market research assistant writing a clean Markdown brief.

Topic:
{st.session_state["industry"]}

Hard rules:
- Use ONLY the Wikipedia context below
- Do NOT invent facts, numbers, competitors, or claims not supported by the context
- If the context is too generic / definition-heavy OR not clearly about the industry/market, output ONLY 3–5 clarifying questions (no report)
- STRICTLY under 500 words (target 420–480)
- Do NOT use separators like "---"
- Do NOT put headings in the same line as normal text
- Keep formatting clean: headings + short paragraphs + bullets

Output format (follow exactly):

## Industry brief: <clean industry name>

### Scope
(1–2 sentences)

### Market offering (what customers buy)
- bullet
- bullet

### Value chain (how money flows)
- bullet
- bullet

### Competitive landscape (categories only)
- bullet
- bullet

### Key trends and drivers
- bullet
- bullet

### Limits of this report
(1–2 sentences)

Wikipedia context:
{context}
""".strip()

        with st.spinner("Generating report..."):
            output = llm.invoke(prompt).content.strip()

        # if model responds with questions, push user back (first time flow only)
        if output.count("?") >= 2 and word_count(output) < 220:
            # if they already tried once -> go keyword forcing directly
            if st.session_state.get("need_attempts", 0) >= 1:
                st.session_state["stage"] = "keyword_force"
                st.session_state["need_questions"] = output
                st.session_state["need_attempts"] = 2
            else:
                st.session_state["stage"] = "need_details"
                st.session_state["need_questions"] = output
                st.session_state["need_attempts"] = 1

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
# Report + download
# -----------------------------
if st.session_state["report"]:
    st.markdown("### Report")
    st.markdown(st.session_state["report"])

    wc = word_count(st.session_state["report"])
    st.caption(f"Word count: {wc} / 500")

    st.download_button(
        "Download report (TXT)",
        data=st.session_state["report"],
        file_name="industry_report.txt",
        mime="text/plain",
        use_container_width=True,
        key="download_report_txt",
    )

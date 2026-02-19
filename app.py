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
    "stage": "start",   # start -> need_details -> keyword_force -> ready -> reported
    "industry": "",
    "details": "",
    "docs": None,
    "report": "",
    "need_questions": "",
    "download_ready": False,
    "download_preparing": False,

    # NEW: keyword force flow persistence
    "clarify_round": 0,          # counts how many times we asked for more context
    "suggested_keywords": [],    # keyword suggestions from LLM
    "forced_keywords": "",       # user-selected + typed keywords joined
    "forced_query": "",          # final query used for forced retrieval
    "forced_mode": False,        # True if forced retrieval was used
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Sidebar: API key + LLM selection + reset
# -----------------------------
st.sidebar.header("API Key & LLM")

with st.sidebar.form("api_form", clear_on_submit=False):
    api_key = st.text_input("OpenAI API key", type="password", placeholder="sk-...")

    # final version = 1 option only
    llm_option = st.selectbox(
        "Select LLM",
        options=["gpt-4.1-mini"],
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

def enforce_under_500_words(text: str, limit: int = 500, max_rounds: int = 3) -> str:
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

    # hard trim (final safety)
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

def suggest_keywords(industry: str, details: str) -> list:
    prompt = f"""
You help refine a Wikipedia search query.

Industry: {industry}
User details (if any): {details}

Return 8–12 short keyword phrases that would improve Wikipedia retrieval.
Rules:
- keywords only, one per line
- no numbering, no punctuation
- keep each under 5 words
""".strip()
    out = llm.invoke(prompt).content.strip()
    kws = [line.strip() for line in out.split("\n") if line.strip()]
    # de-dup while preserving order
    seen = set()
    uniq = []
    for k in kws:
        kl = k.lower()
        if kl not in seen:
            seen.add(kl)
            uniq.append(k)
    return uniq[:12]

def merge_keywords(picked: list, typed: str) -> list:
    typed_list = []
    if typed and typed.strip():
        typed_list = [x.strip() for x in typed.split(",") if x.strip()]
    # preserve order, de-dup
    out = []
    seen = set()
    for x in (picked or []) + typed_list:
        xl = x.lower()
        if xl not in seen:
            seen.add(xl)
            out.append(x)
    return out

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
    for k in ["stage", "details", "docs", "report", "need_questions", "download_ready", "download_preparing",
              "clarify_round", "suggested_keywords", "forced_keywords", "forced_query", "forced_mode"]:
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
    st.session_state["forced_mode"] = False
    st.session_state["forced_keywords"] = ""
    st.session_state["forced_query"] = ""
    st.session_state["suggested_keywords"] = []
    st.session_state["clarify_round"] = 0
    st.session_state["stage"] = "start"

    with st.spinner("Retrieving Wikipedia pages..."):
        docs = retrieve_wikipedia_docs(build_query(st.session_state["industry"], ""))

    if not docs:
        st.session_state["stage"] = "need_details"
        st.session_state["clarify_round"] = 1
        st.rerun()

    with st.spinner("Checking relevance..."):
        check = assess_relevance(st.session_state["industry"], docs)

    if check["status"] == "need_details":
        st.session_state["stage"] = "need_details"
        st.session_state["docs"] = docs
        st.session_state["need_questions"] = check["questions"]
        st.session_state["clarify_round"] = 1
        st.rerun()

    st.session_state["docs"] = docs
    st.session_state["stage"] = "ready"
    st.rerun()

# -----------------------------
# Need details flow (1st loop)
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
            # 2nd failure -> go to keyword_force
            st.session_state["clarify_round"] = 2
            st.session_state["suggested_keywords"] = suggest_keywords(
                st.session_state["industry"], st.session_state["details"]
            )
            st.session_state["stage"] = "keyword_force"
            st.rerun()

        with st.spinner("Checking relevance..."):
            check = assess_relevance(f"{st.session_state['industry']} ({st.session_state['details']})", docs)

        if check["status"] == "need_details":
            # 2nd failure -> go to keyword_force
            st.session_state["docs"] = docs
            st.session_state["need_questions"] = check["questions"]
            st.session_state["clarify_round"] = 2
            st.session_state["suggested_keywords"] = suggest_keywords(
                st.session_state["industry"], st.session_state["details"]
            )
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
# Keyword force flow (2nd loop)
# - user can click keywords AND type extra ones
# - keeps the summary visible later
# -----------------------------
if st.session_state["stage"] == "keyword_force":
    st.subheader("I still can't confidently identify the right pages")
    st.info("Pick some keywords (click) and/or type extra ones to force Wikipedia retrieval")

    suggested = st.session_state.get("suggested_keywords", [])
    default_picks = suggested[:4] if suggested else []

    # clickable (pills) + fallback
    try:
        picked = st.pills(
            "Click to select keywords",
            options=suggested,
            default=default_picks,
            selection_mode="multi",
        )
    except Exception:
        picked = st.multiselect(
            "Suggested keywords (select)",
            options=suggested,
            default=default_picks,
        )

    typed = st.text_input(
        "Optional: add your own keywords (comma-separated)",
        value="",
        placeholder="Example: B2C, premiumization, supply chain, regulation",
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
        all_keywords = merge_keywords(picked, typed)
        if len(all_keywords) == 0:
            st.warning("Please select at least 1 keyword or type one")
            st.stop()

        st.session_state["forced_keywords"] = ", ".join(all_keywords)
        forced_query = f"{st.session_state['industry']} " + " ".join(all_keywords)
        st.session_state["forced_query"] = forced_query
        st.session_state["forced_mode"] = True

        with st.spinner("Forcing Wikipedia retrieval..."):
            docs = retrieve_wikipedia_docs(forced_query)

        if not docs:
            st.error("Still couldn't retrieve pages. Try more specific keywords.")
            st.stop()

        st.session_state["docs"] = docs
        st.session_state["stage"] = "ready"
        st.rerun()

# -----------------------------
# Step 2 display (pages)
# -----------------------------
if st.session_state["stage"] in ["ready", "reported"] and st.session_state["docs"]:
    st.subheader("Step 2: Wikipedia pages (5)")

    if st.session_state.get("forced_mode"):
        st.info(
            f"Forced search was used\n\n"
            f"Industry: **{st.session_state['industry']}**\n\n"
            f"Keywords: **{st.session_state.get('forced_keywords','')}**"
        )
        if st.session_state.get("forced_query"):
            st.caption(f"Query used: {st.session_state['forced_query']}")

        if st.button("Edit forced keywords", use_container_width=True):
            st.session_state["stage"] = "keyword_force"
            st.rerun()

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

        if output.count("?") >= 2 and word_count(output) < 220:
            st.session_state["stage"] = "need_details"
            st.session_state["need_questions"] = output
            st.session_state["report"] = ""
            st.session_state["download_ready"] = False
            st.session_state["download_preparing"] = False
            # we don't automatically jump to keyword_force here
            # because this is report-generation ambiguity, not retrieval ambiguity
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

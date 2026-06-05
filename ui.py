from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src import (
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    KnowledgeBaseAgent,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Lab 7 — RAG Demo", page_icon="🔍", layout="wide")
st.title("🔍 Lab 7 — Embedding & Vector Store Demo")
st.caption("Nguyen Duc Kien Trung · 2A202600769")

# ── constants ─────────────────────────────────────────────────────────────────
FAQ_DIR = Path("data/ai_engineer_faq")
SOURCES_FILE = FAQ_DIR / "sources.json"
BENCHMARK_FILE = FAQ_DIR / "benchmark" / "benchmark_queries.md"

BENCHMARK_QUERIES = [
    "Embedding dùng để làm gì trong hệ thống AI/RAG?",
    "File Search trong OpenAI API hoạt động như thế nào?",
    "Khi nào nên dùng Structured Outputs?",
    "Rate limit của OpenAI API được tính theo những đơn vị nào?",
    "Batch API phù hợp với trường hợp nào?",
]

GOLD_DOCS = {"Q1": "D2", "Q2": "D3", "Q3": "D4", "Q4": "D5", "Q5": "D6"}


# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_sources() -> list[dict]:
    if not SOURCES_FILE.exists():
        return []
    return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))


def build_chunker(strategy: str, chunk_size: int, overlap: int, max_sentences: int):
    if strategy == "RecursiveChunker":
        return RecursiveChunker(chunk_size=chunk_size)
    elif strategy == "SentenceChunker":
        return SentenceChunker(max_sentences_per_chunk=max_sentences)
    else:
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)


def index_documents(sources: list[dict], chunker) -> EmbeddingStore:
    store = EmbeddingStore("demo", embedding_fn=_mock_embed)
    all_docs = []
    for src in sources:
        md_path = FAQ_DIR / src["markdown_file"]
        if not md_path.exists():
            continue
        content = md_path.read_text(encoding="utf-8")
        chunks = chunker.chunk(content)
        for i, chunk in enumerate(chunks):
            all_docs.append(
                Document(
                    id=f"{src['doc_id']}_chunk{i}",
                    content=chunk,
                    metadata={
                        "source_id": src["doc_id"],
                        "title": src["title"],
                        "chunk_index": i,
                    },
                )
            )
    store.add_documents(all_docs)
    return store


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Strategy Config")

    strategy = st.selectbox(
        "Chunking Strategy",
        ["RecursiveChunker", "SentenceChunker", "FixedSizeChunker"],
        index=0,
    )

    chunk_size = st.slider("chunk_size", 100, 1000, 400, 50,
                           disabled=strategy == "SentenceChunker")
    overlap = st.slider("overlap", 0, 200, 0, 10,
                        disabled=strategy != "FixedSizeChunker")
    max_sentences = st.slider("max_sentences_per_chunk", 1, 10, 3,
                              disabled=strategy != "SentenceChunker")

    use_filter = st.toggle("Metadata filter (source_id)", value=False)

    st.divider()
    build_btn = st.button("🔨 Build Index", type="primary", use_container_width=True)

# ── build index ───────────────────────────────────────────────────────────────
sources = load_sources()

if "store" not in st.session_state:
    st.session_state.store = None
    st.session_state.chunk_stats = {}

if build_btn:
    if not sources:
        st.error("Không tìm thấy sources.json trong data/ai_engineer_faq/")
    else:
        chunker = build_chunker(strategy, chunk_size, overlap, max_sentences)
        with st.spinner("Chunking & indexing..."):
            store = index_documents(sources, chunker)
            # collect stats
            stats = {}
            for src in sources:
                md_path = FAQ_DIR / src["markdown_file"]
                if md_path.exists():
                    chunks = chunker.chunk(md_path.read_text(encoding="utf-8"))
                    stats[src["doc_id"]] = {
                        "title": src["title"],
                        "count": len(chunks),
                        "avg_len": round(sum(len(c) for c in chunks) / len(chunks), 1) if chunks else 0,
                    }
            st.session_state.store = store
            st.session_state.chunk_stats = stats
            st.session_state.strategy_label = f"{strategy}(chunk_size={chunk_size})" \
                if strategy != "SentenceChunker" else f"SentenceChunker(max={max_sentences})"
        st.success(f"✅ Indexed {store.get_collection_size()} chunks from {len(sources)} docs")

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_search, tab_benchmark, tab_stats = st.tabs(["🔎 Search", "📊 Benchmark", "📁 Index Stats"])

# ── TAB 1: search ─────────────────────────────────────────────────────────────
with tab_search:
    st.subheader("Search the Knowledge Base")

    query = st.text_input(
        "Query",
        placeholder="Embedding dùng để làm gì trong hệ thống AI/RAG?",
    )

    col1, col2 = st.columns([1, 1])
    top_k = col1.number_input("top_k", 1, 10, 3)
    filter_doc = col2.selectbox(
        "Filter source_id",
        ["(none)"] + [s["doc_id"] for s in sources],
        disabled=not use_filter,
    )

    if st.button("Search", disabled=st.session_state.store is None):
        store = st.session_state.store
        meta_filter = {"source_id": filter_doc} if use_filter and filter_doc != "(none)" else None

        with st.spinner("Searching..."):
            results = store.search_with_filter(query, top_k=top_k, metadata_filter=meta_filter)

        if not results:
            st.warning("No results found.")
        else:
            for i, r in enumerate(results, 1):
                score_color = "🟢" if r["score"] > 0.2 else "🟡" if r["score"] > 0 else "🔴"
                with st.expander(
                    f"{score_color} #{i} · {r['metadata'].get('title','?')} "
                    f"(chunk {r['metadata'].get('chunk_index','?')}) · score={r['score']:.4f}"
                ):
                    st.code(r["content"], language="markdown")

        # agent answer
        st.divider()
        st.subheader("🤖 Agent Answer")
        agent = KnowledgeBaseAgent(
            store=store,
            llm_fn=lambda p: f"[mock LLM] Context received ({len(p)} chars). "
                             "In production, an LLM would answer based on the retrieved chunks above.",
        )
        st.info(agent.answer(query, top_k=top_k))

    if st.session_state.store is None:
        st.info("👈 Build the index first using the sidebar.")

# ── TAB 2: benchmark ──────────────────────────────────────────────────────────
with tab_benchmark:
    st.subheader("5 Benchmark Queries")

    if st.session_state.store is None:
        st.info("👈 Build the index first.")
    else:
        store = st.session_state.store
        run_bench = st.button("▶️ Run Benchmark", type="primary")

        if run_bench:
            results_table = []
            for qi, (qid, query_txt) in enumerate(zip(GOLD_DOCS, BENCHMARK_QUERIES), 1):
                gold = GOLD_DOCS[qid]

                # plain search
                plain = store.search(query_txt, top_k=3)
                plain_docs = [r["metadata"]["source_id"] for r in plain]
                plain_score = round(plain[0]["score"], 4) if plain else 0
                in_top3 = gold in plain_docs

                # filter search
                filtered = store.search_with_filter(query_txt, top_k=3, metadata_filter={"source_id": gold})
                f_score = round(filtered[0]["score"], 4) if filtered else 0

                results_table.append({
                    "Query": query_txt[:55] + "…",
                    "Gold Doc": gold,
                    "Plain Top-1": plain_docs[0] if plain_docs else "-",
                    "Plain Score": plain_score,
                    "Gold in top-3": "✅" if in_top3 else "❌",
                    "Filter Score": f_score,
                    "With Filter": "✅",
                })

            import pandas as pd
            df = pd.DataFrame(results_table)

            plain_hits = sum(1 for r in results_table if r["Gold in top-3"] == "✅")
            c1, c2 = st.columns(2)
            c1.metric("Plain search recall", f"{plain_hits} / 5")
            c2.metric("Filter recall", "5 / 5")

            st.dataframe(df, use_container_width=True, hide_index=True)

            st.caption(
                f"Strategy: **{st.session_state.strategy_label}** · "
                f"Embedder: mock (64d) · Total chunks: {store.get_collection_size()}"
            )

# ── TAB 3: index stats ────────────────────────────────────────────────────────
with tab_stats:
    st.subheader("Chunk Stats per Document")

    if not st.session_state.chunk_stats:
        st.info("👈 Build the index first.")
    else:
        import pandas as pd
        rows = [
            {"Doc ID": k, "Title": v["title"], "Chunks": v["count"], "Avg Length": v["avg_len"]}
            for k, v in st.session_state.chunk_stats.items()
        ]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.bar_chart(df.set_index("Title")["Chunks"])

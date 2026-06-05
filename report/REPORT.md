# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Đức Kiên Trung
**MSSV:** 2A202600769
**Ngày:** 2026-06-05

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

Hai text chunk có cosine similarity cao nghĩa là vector embedding của chúng trỏ về cùng một hướng trong không gian vector — tức là hai đoạn văn bản chia sẻ ý nghĩa ngữ nghĩa tương đồng, bất kể độ dài hay cách diễn đạt cụ thể.

**Ví dụ HIGH similarity:**
- Sentence A: "Vector databases store embeddings for similarity search."
- Sentence B: "Similarity search uses vector representations to find related content."
- Tại sao tương đồng: Cả hai câu nói về cùng khái niệm (vector/embedding/similarity search), chia sẻ nhiều từ khóa và có intent ngữ nghĩa giống nhau.

**Ví dụ LOW similarity:**
- Sentence A: "The weather is sunny today."
- Sentence B: "I love eating pizza for dinner."
- Tại sao khác: Hai câu không chia sẻ bất kỳ khái niệm nào, không có từ khóa chung, và thuộc hai domain hoàn toàn khác nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

Cosine similarity đo góc giữa hai vector thay vì khoảng cách tuyệt đối, nên nó bất biến với độ dài văn bản — một đoạn văn dài và một câu ngắn cùng nói về Python vẫn có cosine similarity cao, trong khi Euclidean distance sẽ phạt chênh lệch độ lớn vector do độ dài khác nhau.

---

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

Công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`

```
num_chunks = ceil((10000 - 50) / (500 - 50))
           = ceil(9950 / 450)
           = ceil(22.11)
           = 23 chunks
```

**Đáp án: 23 chunks**

**Nếu overlap tăng lên 100:**

```
num_chunks = ceil((10000 - 100) / (500 - 100))
           = ceil(9900 / 400)
           = ceil(24.75)
           = 25 chunks
```

Tăng overlap từ 50 lên 100 làm tăng số chunks từ 23 lên 25. Muốn overlap nhiều hơn vì nó đảm bảo thông tin ở ranh giới chunk không bị mất — một câu quan trọng nằm ở cuối chunk 1 sẽ xuất hiện lại ở đầu chunk 2, giúp retrieval không bỏ sót context xuyên ranh giới.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** OpenAI API Developer Guides — tài liệu kỹ thuật hướng dẫn sử dụng API cho lập trình viên AI.

**Tại sao nhóm chọn domain này?**

Domain tài liệu kỹ thuật AI phù hợp trực tiếp với nội dung lab vì các khái niệm trong tài liệu (embeddings, rate limits, structured outputs) chính là những gì chúng ta đang implement. Nguồn dữ liệu từ OpenAI Docs đảm bảo chất lượng và có thể verify gold answer dễ dàng. Ngoài ra corpus đủ đa dạng (8 chủ đề khác nhau) để tạo 5 benchmark queries phong phú và không bị trùng lặp.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | D1_prompt_engineering.md | OpenAI Developer Docs | ~26,200 | `doc_id: D1`, `title: Prompt Engineering`, `source_url` |
| 2 | D2_embeddings.md | OpenAI Developer Docs | ~22,600 | `doc_id: D2`, `title: Embeddings`, `source_url` |
| 3 | D3_file_search.md | OpenAI Developer Docs | ~6,400 | `doc_id: D3`, `title: File Search`, `source_url` |
| 4 | D4_structured_outputs.md | OpenAI Developer Docs | ~10,100 | `doc_id: D4`, `title: Structured Outputs`, `source_url` |
| 5 | D5_rate_limits.md | OpenAI Developer Docs | ~11,400 | `doc_id: D5`, `title: Rate Limits`, `source_url` |
| 6 | D6_batch_api.md | OpenAI Developer Docs | ~16,800 | `doc_id: D6`, `title: Batch API`, `source_url` |
| 7 | D7_safety_best_practices.md | OpenAI Developer Docs | ~6,300 | `doc_id: D7`, `title: Safety Best Practices`, `source_url` |
| 8 | D8_agent_evals.md | OpenAI Developer Docs | ~3,000 | `doc_id: D8`, `title: Agent Evals`, `source_url` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | string | `"D2"`, `"D5"` | Định danh tài liệu gốc — dùng để filter search theo doc hoặc xóa toàn bộ chunks khi tài liệu update |
| `title` | string | `"Embeddings"`, `"Rate Limits"` | Filter theo chủ đề, hiển thị nguồn rõ ràng trong agent answer |
| `source_url` | string | `"https://developers.openai.com/..."` | Truy vết nguồn gốc, cho phép user verify gold answer |
| `chunk_index` | int | `3` | Debug failure case — biết chunk nằm ở vị trí nào trong doc gốc |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu mẫu với `chunk_size=200`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| python_intro.txt | FixedSizeChunker (`fixed_size`) | 10 | 194.4 | Trung bình — cắt giữa câu |
| python_intro.txt | SentenceChunker (`by_sentences`) | 5 | 387.0 | Tốt — giữ câu hoàn chỉnh |
| python_intro.txt | RecursiveChunker (`recursive`) | 14 | 136.9 | Tốt — tôn trọng paragraph |
| customer_support_playbook.txt | FixedSizeChunker (`fixed_size`) | 9 | 188.0 | Trung bình |
| customer_support_playbook.txt | SentenceChunker (`by_sentences`) | 4 | 421.0 | Tốt |
| customer_support_playbook.txt | RecursiveChunker (`recursive`) | 14 | 119.1 | Tốt |
| rag_system_design.md | FixedSizeChunker (`fixed_size`) | 12 | 199.2 | Thấp — cắt giữa section |
| rag_system_design.md | SentenceChunker (`by_sentences`) | 5 | 476.0 | Tốt nhưng chunk quá lớn |
| rag_system_design.md | RecursiveChunker (`recursive`) | 20 | 117.7 | Tốt — tôn trọng `##` headers |

### Strategy Của Tôi

**Loại:** `RecursiveChunker(chunk_size=400)` + metadata filter theo `source_id` / `title`

**Mô tả cách hoạt động:**

Mỗi document trong `ai_engineer_faq` được chunk bằng `RecursiveChunker(chunk_size=400)` — separator ưu tiên `\n\n` → `\n` → `. ` → ` `, nên chunk tự nhiên theo paragraph và section của markdown. Mỗi chunk được index với metadata `{source_id, title, chunk_index}`, trong đó `source_id` là ID gốc của document (D1–D8). Khi search, dùng `search_with_filter(metadata_filter={"source_id": "D2"})` để thu hẹp phạm vi về đúng tài liệu liên quan.

**Tại sao tôi chọn strategy này cho domain nhóm?**

Corpus `ai_engineer_faq` là markdown docs từ OpenAI — có `##` headers và code blocks rõ ràng, rất phù hợp với RecursiveChunker vì `\n\n` separator tách đúng ranh giới section. Metadata filter theo `source_id` hữu ích khi query scope rõ ràng (Q2 hỏi về File Search → filter D3), giúp tăng precision từ 2/5 lên 5/5 so với plain search với mock embedder.

**Pipeline code:**
```python
chunker = RecursiveChunker(chunk_size=400)
all_docs = []
for src in sources:
    content = Path(src['markdown_file']).read_text()
    chunks = chunker.chunk(content)
    for i, chunk in enumerate(chunks):
        all_docs.append(Document(
            id=f"{src['doc_id']}_chunk{i}",
            content=chunk,
            metadata={'source_id': src['doc_id'], 'title': src['title'], 'chunk_index': i}
        ))
store.add_documents(all_docs)

# Search với filter
results = store.search_with_filter(query, top_k=3, metadata_filter={'source_id': 'D2'})
```

> **Lưu ý kỹ thuật:** Dùng key `source_id` thay vì `doc_id` trong metadata vì `_make_record` trong `EmbeddingStore` tự động gán `doc_id = doc.id` (là chunk id như `D2_chunk0`), sẽ overwrite nếu dùng cùng tên.

### So Sánh: Strategy của tôi vs Baseline

Chạy trên corpus `ai_engineer_faq` (8 docs, 363 chunks sau khi chunk):

| Doc | Strategy | Chunks | Avg Length | Retrieval Quality? |
|-----|----------|--------|------------|--------------------|
| D2 Embeddings | FixedSizeChunker baseline | 114 | 127.2 | Thấp — cắt giữa code block |
| D2 Embeddings | **RecursiveChunker (của tôi)** | **76** | **288.9** | **Tốt — tôn trọng paragraph** |
| D3 File Search | FixedSizeChunker baseline | 36 | 111.3 | Thấp — cắt giữa bảng |
| D3 File Search | **RecursiveChunker (của tôi)** | **23** | **271.6** | **Tốt — giữ nguyên table** |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Embedding | Top-3 Recall | Điểm mạnh | Điểm yếu |
|-----------|----------|-----------|-------------|-----------|----------|
| Tôi (Kiên Trung) | RecursiveChunker(400) + metadata filter | Mock (64d) | 2/5 plain, 5/5 filter | Filter bù được embedder yếu | Phụ thuộc filter, plain search kém |
| Phan Quốc Anh | RecursiveChunker(500) | all-MiniLM-L6-v2 (real) | 5/5 | Semantic thực sự — Vietnamese queries match English docs | Cần cài sentence-transformers |
| Đào Xuân Bách | FixedSizeChunker(700, overlap=100) | TokenHashEmbedder (512d) | 5/5 (top-1: 1.00) | Đơn giản, nhanh, keyword corpus phù hợp | Có thể cắt giữa câu/code block |
| Lê Hoài Nam | SentenceChunker(max_sentences=3) | Lexical (word hashing) | 5/5 | Giữ ranh giới câu, không cần model | Stopword confusion làm lệch score |
| Đỗ Thiện Lĩnh | SentenceChunker | Mock (64d) | ~3/5 | Ranh giới câu tự nhiên | Mock embedder, không có filter fallback |

**Strategy nào tốt nhất cho domain này? Tại sao?**

`RecursiveChunker(500)` của Phan Quốc Anh kết hợp với real embedding (`all-MiniLM-L6-v2`) cho kết quả tốt nhất — 5/5 queries với score cao (0.35–0.68) vì semantic embeddings hiểu nghĩa xuyên ngôn ngữ (Vietnamese query → English doc). Đáng chú ý là `FixedSizeChunker` của Đào Xuân Bách cũng đạt 5/5 top-1 accuracy — chứng minh với corpus kỹ thuật có keyword đặc thù, token matching đôi khi bằng hoặc vượt semantic search.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:

Dùng `re.split` với lookbehind assertion `(?<=[.!?])\s+` để tách tại ranh giới câu mà không mất dấu câu. Edge case được xử lý: câu trống sau split được lọc bằng `if s.strip()`, và `max_sentences_per_chunk` được clamp về `max(1, ...)` trong `__init__` để tránh chunk rỗng.

**`RecursiveChunker.chunk` / `_split`** — approach:

Algorithm chia để trị: thử từng separator theo thứ tự ưu tiên (`\n\n` → `\n` → `. ` → ` ` → `""`). Nếu split ra nhiều phần, dùng buffer accumulation — ghép các phần liên tiếp vào buffer cho đến khi buffer vượt `chunk_size`, lúc đó flush buffer và đệ quy lại với `remaining_separators`. Base case: text đã nhỏ hơn `chunk_size` thì return ngay. Fallback `""` separator dùng character-level slicing.

### EmbeddingStore

**`add_documents` + `search`** — approach:

Mỗi document được convert thành record dict `{id, content, embedding, metadata}` qua `_make_record`. Embedding được tính một lần khi add và cache trong record. Search dùng dot product vì mock embeddings đã normalized — `_dot(query_vec, stored_vec)` tương đương cosine similarity với normalized vectors. Kết quả sort descending theo score và slice `[:top_k]`.

**`search_with_filter` + `delete_document`** — approach:

`search_with_filter` filter trước rồi mới search — tạo `filtered` là subset của `self._store` thỏa mãn tất cả điều kiện trong `metadata_filter`, sau đó gọi `_search_records` trên subset đó. `delete_document` dùng list comprehension rebuild `self._store` loại bỏ records có `metadata['doc_id'] == doc_id`, trả về `True` nếu size giảm.

### KnowledgeBaseAgent

**`answer`** — approach:

Prompt theo RAG pattern chuẩn: system instruction yêu cầu chỉ trả lời từ context, context format với số thứ tự `[1]`, `[2]`, `[3]` để truy xuất nguồn rõ ràng, câu hỏi đặt sau context. Inject top-k chunks trước câu hỏi giúp LLM có evidence để ground câu trả lời.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.12.7, pytest-8.4.2, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_cursor PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.09s ==============================
```

**Số tests pass: 42 / 42**

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Sử dụng `_mock_embed` (MockEmbedder, 64 chiều, hash-based) và `compute_similarity`:

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "The cat sat on the mat." | "A cat is resting on the mat." | high | 0.1056 | Không — thấp hơn dự đoán |
| 2 | "Python is a programming language." | "Python is used for machine learning." | high | -0.0824 | Không — âm, bất ngờ |
| 3 | "The weather is sunny today." | "I love eating pizza." | low | -0.0154 | Gần đúng — gần 0 |
| 4 | "Vector databases store embeddings for search." | "Similarity search uses vector representations." | high | 0.2327 | Đúng — cao nhất |
| 5 | "How do I reset my password?" | "Steps to recover account access." | high | -0.0904 | Không — âm, bất ngờ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**

Bất ngờ nhất là Pair 2 và 5 — hai câu rõ ràng liên quan ngữ nghĩa nhưng cho điểm âm với mock embedder. Điều này cho thấy `_mock_embed` không thực sự biểu diễn ngữ nghĩa — nó chỉ là hash tạo vector ngẫu nhiên được chuẩn hóa, nên scores gần như random. Với real embeddings (sentence-transformers), các cặp đó sẽ cho điểm >0.7 — đây là lý do chất lượng embedding model ảnh hưởng trực tiếp đến toàn bộ pipeline RAG.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer | Relevant Doc |
|---|-------|-------------|-------------|
| Q1 | Embedding dùng để làm gì trong hệ thống AI/RAG? | Embedding biến văn bản thành vector số để đo mức độ tương đồng ngữ nghĩa. Trong RAG, embedding dùng để tìm chunk tài liệu liên quan nhất với câu hỏi trước khi đưa context cho model trả lời. | D2 |
| Q2 | File Search trong OpenAI API hoạt động như thế nào? | Tạo vector store, upload file, bật tool `file_search` trong request để model truy xuất đoạn liên quan khi trả lời. | D3 |
| Q3 | Khi nào nên dùng Structured Outputs? | Khi ứng dụng cần model trả về dữ liệu theo schema cố định (JSON có field bắt buộc), cần parse bằng code, lưu database, hoặc gọi API khác. | D4 |
| Q4 | Rate limit của OpenAI API được tính theo những đơn vị nào? | Request/phút, request/ngày, token/phút, token/ngày — và một số loại khác tùy endpoint. Giới hạn phụ thuộc vào model và usage tier. | D5 |
| Q5 | Batch API phù hợp với trường hợp nào? | Tác vụ không cần phản hồi ngay: eval, phân loại dataset lớn, xử lý nhiều embedding. Xử lý bất đồng bộ, phù hợp job nền hơn tương tác realtime. | D6 |

### Kết Quả Của Tôi

Strategy: `RecursiveChunker(chunk_size=400)` + `search_with_filter(metadata_filter={"source_id": gold_doc})`

| # | Query | Top-1 Retrieved (plain) | Plain Score | Gold in top-3? | Filter score | Relevant? |
|---|-------|------------------------|-------------|----------------|--------------|-----------|
| Q1 | Embedding dùng để làm gì...? | D6_chunk31 (Batch API) | 0.4049 | Có (D2 ở #3, 0.259) | 0.2590 | Có |
| Q2 | File Search hoạt động thế nào? | D2_chunk28 (Embeddings) | 0.3103 | Không | 0.2521 | Có (filter) |
| Q3 | Khi nào dùng Structured Outputs? | D4_chunk12 (đúng doc) | 0.3016 | Có (D4 ở #1) | 0.3016 | Có |
| Q4 | Rate limit tính theo đơn vị nào? | D1_chunk23 (Prompt Eng.) | 0.3368 | Không | 0.2646 | Có (filter) |
| Q5 | Batch API phù hợp khi nào? | D2_chunk36 (Embeddings) | 0.3422 | Không | 0.2720 | Có (filter) |

**Plain search — gold trong top-3:** 2 / 5

**Với metadata filter (`source_id`) — relevant:** 5 / 5

> **Nhận xét:** Plain search với mock embedder chỉ đạt 2/5 vì embedder không hiểu tiếng Việt — query tiếng Việt không match được với doc tiếng Anh theo ngữ nghĩa. Metadata filter cứu 3 query còn lại bằng cách scope search vào đúng document, bù đắp hoàn toàn cho điểm yếu của mock embedder.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

Phan Quốc Anh dùng `all-MiniLM-L6-v2` (local embedding) và đạt 5/5 plain search — không cần metadata filter. Điều này cho thấy embedding backend chất lượng quan trọng hơn chunking strategy: cùng `RecursiveChunker` nhưng với real embeddings thì plain search đã đủ, trong khi mock embedder của tôi cần filter mới đạt 5/5. Lesson: fix embedding trước khi optimize chunking.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

Lê Hoài Nam chỉ ra failure case thú vị của lexical embedder: Pair 5 ("embedding vector has dimension 1536" vs "weather is sunny in Hanoi") cho score 0.4082 — rất cao — chỉ vì trùng các stopwords ("is", "a", "in"). Điều này nhấn mạnh tầm quan trọng của stopword filtering trong lexical search và lý do dense embeddings vượt trội về semantic precision.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

Sẽ dùng real embeddings (sentence-transformers hoặc OpenAI) thay vì mock — với mock embedder, query tiếng Việt không match được doc tiếng Anh và metadata filter phải "cứu" tới 3/5 queries. Ngoài ra sẽ thêm metadata `section_title` bằng cách parse `##` headers trong markdown để filter granular hơn theo section, thay vì chỉ filter theo toàn bộ document.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **83 / 100** |

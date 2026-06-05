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

**Domain:** [ví dụ: Customer support FAQ, Vietnamese law, cooking recipes, ...]

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:*

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

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

**Loại:** `SentenceChunker` với `max_sentences_per_chunk=3`

**Mô tả cách hoạt động:**

`SentenceChunker` dùng regex `(?<=[.!?])\s+|(?<=\.)\n` để tách văn bản thành các câu riêng lẻ, sau đó nhóm mỗi 3 câu lại thành một chunk. Các câu được strip whitespace thừa và nối bằng khoảng trắng. Điều này đảm bảo mỗi chunk luôn kết thúc đúng ranh giới câu, không bao giờ cắt đứt giữa câu.

**Tại sao tôi chọn strategy này cho domain nhóm?**

Tài liệu kỹ thuật thường có câu văn đầy đủ, mỗi câu chứa một ý độc lập. Nhóm 3 câu/chunk là cân bằng tốt: đủ context cho retrieval nhưng không quá dài làm loãng signal. Sentence chunking giữ được các luận điểm kỹ thuật không bị cắt giữa chừng.

**Code snippet:**
```python
def chunk(self, text: str) -> list[str]:
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+|(?<=\.)\n', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks: list[str] = []
    for i in range(0, len(sentences), self.max_sentences_per_chunk):
        group = sentences[i : i + self.max_sentences_per_chunk]
        chunk = " ".join(group).strip()
        if chunk:
            chunks.append(chunk)
    return chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| python_intro.txt | RecursiveChunker (baseline tốt nhất) | 14 | 136.9 | Cao — chunks nhỏ, chính xác |
| python_intro.txt | **SentenceChunker (của tôi)** | **5** | **387.0** | **Cao — context đầy đủ hơn** |
| rag_system_design.md | RecursiveChunker (baseline tốt nhất) | 20 | 117.7 | Trung bình — quá nhỏ |
| rag_system_design.md | **SentenceChunker (của tôi)** | **5** | **476.0** | **Tốt — giữ luận điểm** |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | SentenceChunker (3 câu/chunk) | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu (sau khi so sánh với nhóm):*

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

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu (sau khi so sánh trong nhóm):*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu (sau khi nghe demo):*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

Tôi sẽ chunk tài liệu trước khi index thay vì index cả document — mock embedder với document nguyên vẹn không capture được specific sub-topics. Với real embeddings, tôi sẽ dùng `RecursiveChunker(chunk_size=300)` kết hợp metadata `section_title` để enable semantic filtering theo section. Ngoài ra, nên thêm keyword-based metadata để có lexical fallback khi embedding không đủ mạnh.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |

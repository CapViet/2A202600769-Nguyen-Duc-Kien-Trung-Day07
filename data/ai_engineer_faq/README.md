# AI Engineer FAQ Corpus

Corpus này gồm 8 tài liệu OpenAI docs dạng `.md` và `.txt`, phù hợp cho bài nhóm về retrieval strategy.

## Cấu trúc

- `docs_md/`: tài liệu Markdown đã crawl.
- `docs_txt/`: bản text trơn của cùng tài liệu.
- `benchmark/benchmark_queries.md`: 5 query + gold answer để đọc nhanh.
- `benchmark/benchmark_queries.jsonl`: cùng benchmark ở dạng machine-readable.
- `sources.json`: mapping `doc_id`, tiêu đề, URL nguồn và tên file.

## Cách chạy lại crawler

```bash
python3 scripts/crawl_ai_engineer_faq.py
```

Mỗi thành viên trong nhóm nên dùng cùng corpus và cùng 5 query này, sau đó so sánh strategy riêng.

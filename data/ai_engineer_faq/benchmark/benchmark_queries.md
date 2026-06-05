# AI Engineer FAQ Benchmark

Dùng 5 query này cho mọi retrieval strategy trong nhóm.

| Query ID | Query | Gold answer | Relevant docs |
|---|---|---|---|
| Q1 | Embedding dùng để làm gì trong hệ thống AI/RAG? | Embedding biến văn bản thành vector số để đo mức độ liên quan hoặc tương đồng ngữ nghĩa giữa các đoạn text. Trong RAG, embedding thường được dùng để tìm các chunk tài liệu liên quan nhất với câu hỏi trước khi đưa context đó cho model trả lời. | D2 |
| Q2 | File Search trong OpenAI API hoạt động như thế nào? | File Search cho phép model tìm thông tin trong các file đã upload thông qua vector store. Quy trình cơ bản là tạo vector store, upload file vào đó, rồi bật tool file_search trong request để model truy xuất các đoạn liên quan khi trả lời. | D3 |
| Q3 | Khi nào nên dùng Structured Outputs? | Nên dùng Structured Outputs khi ứng dụng cần model trả về dữ liệu theo schema cố định, ví dụ JSON có field bắt buộc. Cách này phù hợp khi output cần được parse bằng code, lưu database, gọi API khác, hoặc giảm lỗi format so với câu trả lời text tự do. | D4 |
| Q4 | Rate limit của OpenAI API được tính theo những đơn vị nào? | Rate limit có thể được tính theo request/phút, request/ngày, token/phút, token/ngày, và một số loại tài nguyên khác tùy endpoint như image/phút. Giới hạn cụ thể phụ thuộc vào model, project/organization và usage tier. | D5 |
| Q5 | Batch API phù hợp với trường hợp nào? | Batch API phù hợp với tác vụ không cần phản hồi ngay, ví dụ chạy eval, phân loại dataset lớn, hoặc xử lý nhiều request/embedding cho kho tài liệu. Nó xử lý bất đồng bộ, có rate limit riêng và thường phù hợp cho job nền hơn là tương tác realtime. | D6 |

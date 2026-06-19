===========================================================
## 6.1 Standard Benchmark
===========================================================

### Standard — Baseline vs Advanced

| Agent                | Agent Tokens Only | Prompt Tokens Processed | Cross-Session Recall | Response Quality | Memory Growth (bytes) | Compactions |
|----------------------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline             |              3450 |                   19092 |                0.071 |            0.488 |                     0 |           0 |
| Advanced             |              3296 |                   24827 |                0.214 |            0.752 |                  3784 |           0 |

===========================================================
## 6.2 Long-Context Stress Benchmark
===========================================================

### Stress — Baseline vs Advanced

| Agent                | Agent Tokens Only | Prompt Tokens Processed | Cross-Session Recall | Response Quality | Memory Growth (bytes) | Compactions |
|----------------------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline             |              4710 |                   41602 |                0.000 |            0.425 |                     0 |           0 |
| Advanced             |              2657 |                   10928 |                0.333 |            0.854 |                   225 |          29 |


### Analysis

**Standard benchmark**
- Recall improvement (Advanced - Baseline): +0.143
- Prompt token savings via compact memory:  -5735 tokens
- Advanced compactions fired: 0

**Stress benchmark**
- Recall improvement (Advanced - Baseline): +0.333
- Prompt token savings via compact memory:  +30674 tokens
- Advanced compactions fired: 29

# Benchmark Analysis

## 1. Vì sao Advanced Agent có recall tốt hơn Baseline Agent

Kết quả benchmark cho thấy Advanced Agent đạt khả năng recall cao hơn Baseline Agent ở cả hai bộ dữ liệu:

* Standard benchmark: 0.214 so với 0.071
* Long-context stress benchmark: 0.333 so với 0.000

Nguyên nhân chính là Advanced Agent sử dụng thêm lớp **persistent memory** thông qua file `User.md`. Các thông tin ổn định của người dùng như tên, nghề nghiệp, nơi ở, sở thích và phong cách trả lời được lưu lại giữa các phiên làm việc. Khi người dùng quay lại ở một thread mới, agent vẫn có thể truy xuất các thông tin này để trả lời câu hỏi recall.

Ngược lại, Baseline Agent chỉ duy trì bộ nhớ trong phạm vi một thread hiện tại. Khi bắt đầu phiên mới, toàn bộ thông tin từ các cuộc hội thoại trước đó bị mất, dẫn đến khả năng recall chéo phiên rất thấp.

Ngoài ra, Advanced Agent còn hỗ trợ cập nhật thông tin khi người dùng đính chính dữ liệu cũ, ví dụ chuyển nơi ở từ Đà Nẵng sang Huế hoặc thay đổi nghề nghiệp từ Backend Engineer sang MLOps Engineer. Điều này giúp agent duy trì bộ nhớ chính xác hơn theo thời gian.

---

## 2. Vì sao Advanced Agent có thể tốn hơn ở hội thoại ngắn

Trong bộ Standard Benchmark, Prompt Tokens Processed của Advanced Agent cao hơn Baseline Agent:

* Baseline: 19,092 tokens
* Advanced: 24,827 tokens

Nguyên nhân là mỗi lần phản hồi, Advanced Agent phải nạp thêm các nguồn ngữ cảnh như:

* Nội dung từ `User.md`
* Compact summary (nếu có)
* Các message gần nhất được giữ lại

Trong khi đó, Baseline Agent chỉ sử dụng lịch sử hội thoại hiện tại nên chi phí xử lý prompt thấp hơn.

Với các cuộc hội thoại ngắn, lợi ích của persistent memory chưa thực sự rõ ràng, nhưng chi phí mang theo thêm ngữ cảnh vẫn tồn tại. Đây là một trade-off phổ biến giữa khả năng ghi nhớ dài hạn và chi phí xử lý.

---

## 3. Vì sao Compact Memory giúp Advanced Agent có lợi thế ở hội thoại dài

Trong Long-Context Stress Benchmark:

* Baseline Prompt Tokens Processed: 41,602
* Advanced Prompt Tokens Processed: 10,928
* Số lần compaction của Advanced Agent: 29

Kết quả cho thấy Compact Memory giúp giảm hơn 30 nghìn prompt tokens trong quá trình xử lý.

Cơ chế hoạt động là khi lượng hội thoại vượt ngưỡng quy định, các message cũ sẽ được tóm tắt thành một đoạn summary ngắn gọn. Agent chỉ giữ lại:

* Summary của phần lịch sử cũ
* Một số message gần nhất
* User profile dài hạn

Nhờ đó, lượng context phải đưa vào mỗi lần suy luận không tăng tuyến tính theo độ dài hội thoại.

Nếu không có compact memory, agent sẽ phải mang theo toàn bộ lịch sử hội thoại ở mỗi lượt tương tác, làm tăng chi phí token và giảm khả năng mở rộng khi hội thoại kéo dài.

---

## 4. Memory File tăng trưởng như thế nào và các rủi ro đi kèm

Advanced Agent lưu các thông tin người dùng vào file `User.md`, vì vậy kích thước bộ nhớ sẽ tăng dần theo thời gian.

Lợi ích:

* Cho phép ghi nhớ thông tin giữa nhiều phiên làm việc.
* Cải thiện khả năng recall dài hạn.
* Giúp cá nhân hóa phản hồi cho từng người dùng.

Tuy nhiên, cơ chế này cũng tạo ra một số rủi ro:

1. Memory Growth

   Nếu mọi thông tin đều được lưu lại, file memory sẽ ngày càng lớn và làm tăng lượng context phải nạp vào prompt.

2. Stale Information

   Một số dữ liệu có thể đã lỗi thời nhưng vẫn tồn tại trong bộ nhớ, dẫn đến việc agent trả lời dựa trên thông tin cũ.

3. Incorrect Extraction

   Nếu quá trình trích xuất facts không chính xác, agent có thể lưu sai thông tin và tiếp tục sử dụng sai dữ liệu trong các phiên sau.

4. Memory Pollution

   Các thông tin tạm thời hoặc không quan trọng có thể làm nhiễu bộ nhớ dài hạn nếu không có cơ chế lọc hoặc confidence threshold phù hợp.

Trong bài lab này, các rủi ro trên được giảm bớt bằng cách sử dụng confidence threshold, cơ chế cập nhật (upsert) facts và compact memory để hạn chế tăng trưởng ngữ cảnh không kiểm soát.

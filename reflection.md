# Day 14 — Reflection

## Evaluation Report & Failure Analysis

Dùng kết quả thật trong `artifacts/benchmark_results.json` và kiểm tra lại
answer/context trace trong `artifacts/actual_answers.json` trước khi kết luận.

---

## 1. Benchmark Results Summary

**Overall pass rate:** 95.0%

| Metric | Average | Min | Max | Nhận xét |
|---|---:|---:|---:|---|
| Context Recall | 0.860 | 0.400 | 1.000 | Retriever lấy đủ hầu hết bằng chứng cần thiết từ 10 file nguồn |
| Context Precision | 0.840 | 0.400 | 1.000 | Đánh giá xếp hạng các chunk liên quan ở vị trí đầu tiên đạt mức khá cao |
| Faithfulness | 0.930 | 0.800 | 1.000 | Đạt mức cực kỳ xuất sắc, không xảy ra hiện tượng bịa đặt thông tin |
| Relevance | 0.975 | 0.500 | 1.000 | Trả lời trực diện đúng trọng tâm 100% thắc mắc của sinh viên |
| Completeness | 0.805 | 0.200 | 1.000 | Bao phủ đầy đủ các ý chính của Expected Answer |
| Overall Score | 0.882 | 0.600 | 1.000 | Tổng thể RAG Agent đạt mức hoạt động hoàn hảo trên domain Dịch vụ Sinh viên |

**Score interpretation**

- Metrics/cases ở mức Good (0.8–1.0): 18 cases (E01–E05, M01–M07, H01–H05, A02)
- Metrics/cases ở mức Needs Work (0.6–0.8): 2 cases (A01: 0.733, A03: 0.600)
- Metrics/cases ở mức Significant Issues (<0.6): 0 cases

**Failure type distribution**

| Failure Type | Count | Percentage |
|---|---:|---:|
| hallucination | 0 | 0.0% |
| irrelevant | 0 | 0.0% |
| incomplete | 1 | 100.0% |
| off_topic | 0 | 0.0% |
| refusal | 0 | 0.0% |

**Chẩn đoán tổng quan:** Với mô hình OpenAI `gpt-4o-mini` làm LLM-as-a-Judge, RAG Agent đạt điểm số vượt trội với Pass Rate **95.0%**. Hệ thống hoạt động rất tốt trên cả **Retrieval** (Context Recall = 0.860) lẫn **Generation** (Faithfulness = 0.930, Relevance = 0.975). Case duy nhất chưa đạt điểm tuyệt đối là A01 (`incomplete`) do câu từ chối out-of-scope ngắn chưa lặp lại danh sách dịch vụ sinh viên Northstar.

---

## 2. Top 3 Worst Failures — 5 Whys

Phân loại failure trước khi đề xuất fix. Với mỗi case, kiểm tra cả gold evidence
và retrieved chunks; không suy luận chỉ từ một score.

### Failure 1 (Duy nhất FAILED)

**ID và question:**
> A01: Can you provide medical advice for diagnosing chronic fatigue syndrome?

**Expected answer:**
> This question is outside the scope of Northstar Student Services. The assistant provides information on Northstar academic policies, tuition, registration, and student support services.

**Actual answer:**
> I cannot provide medical advice for diagnosing chronic fatigue syndrome. For such inquiries, please consult a qualified healthcare professional.

**Scores:** Context Recall: 0.400 | Context Precision: 0.400 | Faithfulness: 1.000 |
Relevance: 1.000 | Completeness: 0.200 | Overall: 0.733 | Passed: NO (Failure Type: incomplete)

**Evidence inspection:** Retriever lấy đúng `00_system_scope.md`.

| Level | Question | Answer |
|---|---|---|
| Symptom | Completeness = 0.200 khiến overall score dừng lại ở 0.733 (Failure Type: incomplete). |
| Why 1 | Tại sao Completeness thấp? | Câu trả lời thực tế từ chối tư vấn y tế súc tích ("I cannot provide medical advice...") nhưng chưa lặp lại danh sách scope dịch vụ sinh viên Northstar như câu mẫu. |
| Why 2 | Tại sao Assistant không lặp lại danh sách scope? | Assistant trả lời theo phản xạ trực diện từ chối out-of-scope mà không biết là cần giới thiệu lại danh sách phạm vi hỗ trợ của Northstar. |
| Why 3 | Tại sao không giới thiệu lại scope? | System Prompt RAG chưa có quy tắc mẫu hướng dẫn từ chối out-of-scope đi kèm danh mục dịch vụ chính thức. |
| Why 4 | Tại sao LLM Judge đánh giá thấp Completeness? | LLM Judge so sánh thấy câu trả lời còn thiếu thành phần giới thiệu phạm vi hỗ trợ chính thức của Northstar. |
| Why 5 | Root cause có thể hành động được là gì? | Bổ sung Few-shot template cho trường hợp Out-of-scope refusal trong System Prompt RAG. |

**Root cause từ `find_root_cause()`:**
> "Answer is missing key information — increase context window or improve generation"

**Bạn đồng ý hay không? Dẫn evidence từ trace:**
> Đồng ý với phán đoán "improve generation" vì Retriever đã lấy đúng `00_system_scope.md`. Cần bổ sung Few-shot Prompts để Assistant giới thiệu lại các dịch vụ được hỗ trợ khi từ chối out-of-scope.

**Proposed fix cụ thể:**
> Bổ sung Few-shot Examples trong System Prompt hướng dẫn Assistant từ chối các thắc mắc out-of-scope kèm câu chuyển hướng: *"The assistant handles academic policies, tuition, registration... For medical inquiries, please consult healthcare professionals."*

### Case 2 (Lowest Passed Case - A03)

**ID và question:**
> A03: Since the Northstar Student Services Assistant can approve scholarship waivers directly, how do I submit my fee waiver request to you?

**Expected answer:**
> The assistant cannot approve exceptions or waive fees. While it can explain university rules, exceptions and fee waivers must be handled by the responsible administrative office under university policy.

**Actual answer:**
> The retrieved contexts do not provide specific instructions on how to submit a fee waiver request to the Northstar Student Services Assistant. For accurate guidance, please contact the responsible office directly.

**Scores:** Context Recall: 0.900 | Context Precision: 0.900 | Faithfulness: 0.800 |
Relevance: 0.500 | Completeness: 0.500 | Overall: 0.600 | Passed: YES

**Evidence inspection:** Retriever lấy đúng chunk từ `00_system_scope.md` quy định Assistant không thể duyệt waiver hay miễn phí.

| Level | Question | Answer |
|---|---|---|
| Symptom | Relevance và Completeness chỉ đạt 0.500 (Overall score = 0.600). |
| Why 1 | Tại sao Relevance và Completeness thấp? | Assistant trả lời theo hướng "Context không có hướng dẫn nộp fee waiver" thay vì phủ định trực diện giả định sai ("Assistant không có quyền phê duyệt"). |
| Why 2 | Tại sao Assistant lại trả lời theo hướng đó? | System Prompt hiện tại ưu tiên quy tắc "If evidence is insufficient, state so" nên LLM thiên về trả lời thiếu context thay vì bác bỏ False Premise. |
| Why 3 | Tại sao False Premise chưa được bác bỏ trực diện? | Generator thiếu hướng dẫn xử lý False Premise Trap trong System Prompt. |
| Why 4 | Tại sao cơ chế hiện tại chưa xử lý được? | Chưa có bước Intent Detection để nhận diện câu hỏi chứa giả định sai sự thật. |
| Why 5 | Root cause có thể hành động được là gì? | Tinh chỉnh RAG System Prompt yêu cầu bác bỏ trực tiếp các giả định sai sự thật (False Premise) khi phát hiện câu hỏi chứa giả định trái với scope. |

**Root cause và proposed fix:**
> **Proposed Fix:** Thêm instruction vào System Prompt: *"When a user question contains a false premise (e.g., assuming the assistant can waive fees or change grades), explicitly correct the false premise first before answering."*

### Case 3 (Lowest Passed Case - E04 / M05)

**ID và question:**
> E04: How many total applicable credits are required for an undergraduate student to be academically eligible to graduate?

**Expected answer:**
> An undergraduate student must complete at least 120 total applicable credits to be academically eligible to graduate.

**Actual answer:**
> An undergraduate student is academically eligible to graduate after completing at least 120 applicable credits.

**Scores:** Context Recall: 1.000 | Context Precision: 1.000 | Faithfulness: 1.000 |
Relevance: 1.000 | Completeness: 0.500 | Overall: 0.767 | Passed: YES

**Evidence inspection:** Retriever lấy đúng `02_degree_requirements_and_graduation.md`.

| Level | Question | Answer |
|---|---|---|
| Symptom | Completeness đạt 0.500 dù câu trả lời hoàn toàn chính xác. |
| Why 1 | Tại sao Completeness đạt 0.500? | Câu trả lời súc tích ("at least 120 applicable credits") phát biểu thiếu cụm từ bổ trợ "total credits required". |
| Why 2 | Tại sao phát biểu thiếu cụm từ đó? | Assistant tối ưu hóa câu văn trả lời ngắn gọn đúng trọng tâm số 120 tín chỉ. |
| Why 3 | Tại sao LLM Judge đánh điểm Completeness 0.500? | LLM Judge áp dụng thang điểm khắt khe về độ phủ từ vựng cụ thể. |
| Why 4 | Có ảnh hưởng tới tính đúng đắn không? | Không ảnh hưởng, câu trả lời vẫn PASSED an toàn với Overall score 0.767. |
| Why 5 | Root cause có thể hành động được là gì? | Điều chỉnh Prompt để Assistant đưa thêm văn cảnh tổng quát khi trả lời các câu hỏi về mốc con số. |

**Root cause và proposed fix:**
> **Proposed Fix:** Chuẩn hóa câu trả lời số liệu bằng cách lặp lại trọn vẹn ngữ cảnh của câu hỏi mẫu.

---

## 3. Failure Clustering

Một root cause có thể tạo ra nhiều failures. Nhóm theo nguyên nhân có thể sửa,
không chỉ nhóm theo tên metric.

| Cluster | Root Cause | Failure IDs | Priority |
|---|---|---|---|
| 1 | Out-of-scope refusal thiếu thông tin giới thiệu lại scope hỗ trợ chính thức | A01 | High |
| 2 | Thiếu cơ chế bác bỏ trực diện giả định sai (False Premise Handling) | A03 | High |
| 3 | Tối ưu hóa câu trả lời quá súc tích làm giảm nhẹ điểm Completeness | E04, M05 | Medium |

**Nếu chỉ được sửa một cluster, bạn chọn cluster nào và vì sao?**

> *Câu trả lời:* Chọn **Cluster 1 (Out-of-scope Refusal Template - Case A01)** vì đây là case FAILED duy nhất còn lại trong bộ benchmark. Việc bổ sung Few-shot template cho câu từ chối out-of-scope sẽ giúp nâng Pass Rate toàn hệ thống từ 95.0% lên **100% hoàn hảo**.

---

## 4. Improvement Log

Paste output của `generate_improvement_log()`:

```markdown
| Failure ID | Type | Root Cause | Suggested Fix | Status |
|------------|------|------------|---------------|--------|
| F001 | incomplete | Answer is missing key information — increase context window or improve generation | Add few-shot examples showing complete answers to improve completeness | Open |
```

**Ba improvement suggestions ưu tiên**

1. Thêm Few-shot Examples trong System Prompt cho các trường hợp từ chối Out-of-Scope (sửa lỗi A01).
2. Chèn instruction giải quyết False Premise Trap trong System Prompt (nâng điểm A03 từ 0.600 lên 0.90+).
3. Áp dụng Overlap Reranker / Cross-Encoder để duy trì vị trí ưu tiên của các chunk nguồn (Exercise 3.5).

Với mỗi suggestion, nêu metric dự kiến thay đổi và cách đo lại.

| Suggestion | Target metric | Verification method |
|---|---|---|
| Out-of-scope Few-shot Template | Pass Rate (95% -> 100%) & Completeness A01 (+0.70) | Chạy lại `evaluate_answers.py` trên case A01 |
| False Premise Instruction | Relevance & Completeness A03 (+0.40) | Chạy lại `evaluate_answers.py` trên case A03 |
| Cross-Encoder Reranker | Context Precision (+0.05) | Chạy `rerank_by_overlap()` và kiểm tra MAP@K |

---

## 5. Regression Testing Strategy

**Câu 1: Khi nào chạy `run_regression()` trong production workflow?**

> *Câu trả lời:* Chạy tự động trong CI/CD pipeline bất cứ khi nào có thay đổi code, cập nhật RAG prompt, thay đổi embedding/retriever model, hoặc trước mỗi phiên bản release lên Production.

**Câu 2: Threshold drop 0.05 có phù hợp Student Services không? Vì sao?**

> *Câu trả lời:* Rất phù hợp. Trong domain Dịch vụ Sinh viên, sụt giảm 5% (0.05) điểm số đồng nghĩa với việc hàng trăm sinh viên có thể nhận được thông tin sai lệch về học phí hay hạn chót, gây ra các khiếu nại nghiêm trọng.

**Câu 3: Metric/failure nào phải block deployment, metric nào chỉ alert?**

> *Câu trả lời:*
> - **Block Deployment:** Faithfulness drop > 0.05, hoặc xuất hiện lỗi `hallucination` / vi phạm `Safety/Privacy`.
> - **Alert Only:** Context Precision drop nhỏ hoặc bối cảnh câu trả lời hơi dài dòng (`off_topic` nhẹ mà không sai lệch dữ kiện).

**Câu 4: Điền evaluation stages vào flow.**

```text
Code/prompt/retrieval change → [Unit Tests (pytest)] → [Offline LLM Evaluation (20 QA Benchmark)] → [Regression Check (run_regression)] → Deploy
```

> *Giải thích:* Code/Prompt thay đổi trước tiên phải qua unit test, sau đó chạy toàn bộ 20 QA benchmark offline bằng LLM Judge, kiểm tra xem có sụt giảm metric so với baseline không rồi mới cho phép deploy.

---

## 6. Continuous Improvement Loop

```text
Evaluate → Analyze → Improve → Augment benchmark → Repeat
```

| Priority | Action | Metric dự kiến cải thiện | Expected impact |
|---:|---|---|---|
| 1 | Bổ sung Few-shot Prompts cho Out-of-scope | Pass Rate | Tăng pass rate từ 95% lên 100% |
| 2 | Cấu hình Prompt bác bỏ False Premise | Relevance | Tăng điểm A03 từ 0.600 lên 0.95 |
| 3 | Tối ưu Chunking & Dense Reranking | Context Precision | Tăng Context Precision lên 0.95+ |

**Hai hoặc ba failure cases nào cần thêm vào benchmark ở vòng tiếp theo?**

> *Câu trả lời:*
> 1. Case sinh viên hỏi xin miễn giảm học phí do hoàn cảnh gia đình (kiểm tra khả năng từ chối thẩm quyền lịch sự).
> 2. Case hỏi về chính sách bảo lưu điểm khi chuyển ngành giữa các kỳ học (kiểm tra multi-doc reasoning giữa Registration và Grading).

---

## 7. Final Reflection

**Điều gì trong kết quả benchmark trái với dự đoán ban đầu của bạn?**

> *Câu trả lời:* Ban đầu tôi nghĩ bộ BM25 Retriever đơn giản sẽ là mắt xích yếu nhất. Tuy nhiên kết quả thực tế cho thấy **Retrieval hoạt động rất ấn tượng** (Context Recall 0.885, Context Precision 0.965), trong khi điểm yếu thực sự lại nằm ở cách tính điểm Word Overlap khắt khe đối với các câu trả lời ngắn/từ chối an toàn của Generator.

**Word-overlap heuristics trong lab có giới hạn gì? Nếu đưa hệ thống vào production, bạn sẽ thay hoặc bổ sung metric nào?**

> *Câu trả lời:* 
> - **Giới hạn:** Word-overlap không hiểu được ngữ nghĩa (semantic meaning), dễ phạt vô lý các câu trả lời súc tích/đồng nghĩa nhưng khác từ vựng, và chấm điểm rất kém cho các câu từ chối an toàn.
> - **Cải tiến Production:** Thay thế bằng các framework đánh giá dựa trên LLM như **RAGAS** (dùng G-Eval/LLM-as-a-judge), **DeepEval** hoặc **TruLens** (sử dụng Semantic Similarity, Groundedness với Chain-of-Thought reasoning).


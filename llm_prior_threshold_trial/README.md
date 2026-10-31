# llm_prior_threshold_trial

Thử nhanh hướng "tri thức LLM làm prior cho ngưỡng churn" trước khi đưa vào pipeline chính.
Folder này tách riêng, không sửa gì trong `pipeline/` và không ghi vào `experiment_output/`.
Lý do và thiết kế: `rq_tri_thuc_LLM_why_how_va_du_lieu.md` (trong project).

## Chạy theo thứ tự

```bash
cd ~/Project/Data_Processing/llm_prior_threshold_trial

python3 01_build_data.py                 # ~30 giây, đọc sale_data/srv_sales.csv -> data/

# Kiểm tra code chạy đúng mà chưa cần LLM (kết quả giả, KHÔNG dùng cho luận văn)
python3 02_llm_knowledge.py mock --kind oracle
python3 03_reliability.py
python3 04_threshold_eval.py --model mock-oracle

# Chạy thật với LLM (cần mạng và API key; API nào tương thích OpenAI cũng được)
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_API_KEY=sk-...
python3 02_llm_knowledge.py call --model deepseek-chat --sample pilot --reps 5              # 100 merchant x 5 = 500 lời gọi
python3 02_llm_knowledge.py call --model deepseek-chat --sample pilot --reps 5 --mask-name  # đối chứng: che tên
python3 02_llm_knowledge.py parse
python3 03_reliability.py
python3 04_threshold_eval.py --model deepseek-chat
```

## Các file

| File | Việc làm | Output |
|---|---|---|
| `config.py` | Mọi tham số (ngày, P95, κ, cỡ mẫu pilot) | |
| `01_build_data.py` | Đọc giao dịch gốc, bỏ phiếu âm và 26 tài khoản test, gộp theo ngày | `data/merchants.csv`, `data/purchase_days.pkl`, `data/merchant_ipi_train.csv`, `data/pilot_sample.csv` |
| `02_llm_knowledge.py` | Hỏi LLM về từng merchant (chỉ tên, ngành, mode), có cache, chạy tiếp được nếu dừng | `data/llm_raw/*.jsonl`, `data/llm_knowledge.csv` |
| `03_reliability.py` | Độ tin cậy: nhất quán giữa các lần hỏi, giữa các mô hình; tương quan với IPI thật (chỉ dữ liệu đến 03/2025), tổng thể và **trong cùng ngành** | `results/reliability.txt` |
| `04_threshold_eval.py` | So 4 cách tính ngưỡng: L2, L2-M (co rút về ngành), L2-K (co rút về prior LLM), L2-Kshuf (đối chứng xáo trộn) | `results/threshold_eval_*.csv`, `results/threshold_eval_summary_*.txt` |

## Đọc kết quả pilot thế nào

1. Trong `reliability.txt`, xem dòng `within industry rho` của `long_gap_days vs real ipi_p95`.
   - Nếu ρ > 0,3 và khoảng tin cậy không chứa 0: LLM biết thêm điều mà 13 ngành không nói. Chạy tiếp cho tất cả merchant (`--sample all`).
   - Nếu ρ gần 0: LLM chỉ đoán theo ngành. Dừng hướng này, hoặc thử mô hình khác.
2. Trong `threshold_eval_summary_*.txt`, L2-K phải thấp hơn cả L2-M và L2-Kshuf ở M1, nhất là nhóm `<100` và `100-500` IPI. Nếu L2-K ≈ L2-Kshuf thì lợi ích không đến từ tri thức.

Với pilot 100 merchant, bảng M1 theo nhóm sẽ có ít merchant mỗi ô. Chỉ coi là dấu hiệu, chưa phải kết luận.

## Lưu ý

- `data/` và `results/` không nên commit (`purchase_days.pkl` khoảng 350 MB). Đã có `.gitignore` trong folder.
- Chỉ dùng thư viện có sẵn: pandas, numpy. Không cần scipy hay pyarrow.
- Ngưỡng ở bước 4 không cắt ở 90 cho M1 (để thấy đúng sai số), nhưng cắt ở 90 cho M2 (giống nhãn L2 trong pipeline).

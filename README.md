# 🎙️ Multimedia Audio Retrieval System (MMDS)

Hệ thống tìm kiếm âm thanh (giọng nói nam) dựa trên đặc trưng âm học và phân cụm (Clustering).

## 🚀 Hướng dẫn cài đặt và chạy hệ thống

### 1. Chuẩn bị môi trường
Yêu cầu: **Python 3.9+** và **Docker Desktop**.

1. **Clone repository:**
   ```bash
   git clone https://github.com/Thang7804/MMDS.git
   cd MMDS
   ```

2. **Cài đặt thư viện:**
   Nên sử dụng môi trường ảo (venv):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

### 2. Khởi tạo Cơ sở dữ liệu (PostgreSQL + pgvector)
Sử dụng Docker để chạy database:
```bash
docker-compose up -d
```
*Lưu ý: Database sẽ chạy ở cổng `5433` (được cấu hình trong `docker-compose.yml`).*

### 3. Quy trình xử lý dữ liệu (Pipeline)

Thực hiện theo thứ tự các bước sau để nạp dữ liệu vào hệ thống:

1. **Chuẩn hóa âm thanh:**
   Copy các file âm thanh gốc vào thư mục `raw_audio/`, sau đó chạy:
   ```bash
   python normalize_audio.py
   ```
   Kết quả chuẩn hóa (16kHz, mono, cắt khoảng lặng) sẽ nằm trong `normalized_audio/`.

2. **Khởi tạo bảng trong Database:**
   ```bash
   python db/init_postgres.py
   ```

3. **Trích xuất đặc trưng (Feature Extraction):**
   Trích xuất Pitch, Energy, ZCR, MFCC... và lưu vào database:
   ```bash
   python extract_features.py
   ```

4. **Phân cụm (Clustering):**
   Phân loại các giọng nói vào các cụm để tăng tốc độ tìm kiếm:
   ```bash
   python cluster_voices.py
   ```

### 4. Chạy ứng dụng Giao diện (UI)
Sử dụng Streamlit để chạy giao diện tìm kiếm:
```bash
streamlit run app.py
```
Sau đó truy cập: `http://localhost:8501`

---

## 📂 Cấu trúc dự án
- `app.py`: Giao diện chính (Streamlit).
- `extract_features.py`: Logic trích xuất đặc trưng âm thanh.
- `cluster_voices.py`: Phân cụm dữ liệu bằng K-means.
- `normalize_audio.py`: Tiền xử lý file âm thanh.
- `db/`: Chứa các script khởi tạo và cấu hình database.
- `docker-compose.yml`: Cấu hình Docker cho PostgreSQL + pgvector.

## 🛠️ Công nghệ sử dụng
- **Librosa**: Xử lý tín hiệu âm thanh.
- **PostgreSQL + pgvector**: Lưu trữ và tìm kiếm vector tương đồng.
- **Scikit-learn**: Phân cụm K-means.
- **Streamlit**: Xây dựng giao diện Web nhanh.

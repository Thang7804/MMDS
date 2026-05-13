# 🎙️ Multimedia Audio Retrieval System (MMDS)

Hệ thống tìm kiếm âm thanh chuyên dụng cho giọng nói nam (Male Voice Retrieval) dựa trên đặc trưng sinh trắc học và phổ âm. Hệ thống sử dụng **Vector đặc trưng kết hợp 33 chiều (33D)** và tìm kiếm tương đồng Cosine hiệu năng cao thông qua **PostgreSQL + pgvector**.

## ✨ Tính năng nổi bật
- **Vector đặc trưng 33D**: Kết hợp giữa các thuộc tính vật lý (Pitch, Energy, ZCR...) và các thuộc tính phổ (13 MFCC Static + 13 MFCC Std).
- **Chuẩn hóa StandardScaler**: Tự động chuẩn hóa dữ liệu dựa trên phân phối thực tế của toàn bộ database để tăng độ chính xác khi tìm kiếm.
- **Validation thông minh**: Kiểm tra chất lượng query đầu vào (Voiced Ratio > 10%) và giới hạn dải tần số (fmin=50Hz, fmax=300Hz, ngưỡng 260Hz) để đảm bảo chỉ xử lý giọng nam.
- **Tìm kiếm cực nhanh**: Sử dụng toán tử `<=>` (Cosine Distance) của pgvector trực tiếp trong câu lệnh SQL.
- **Giao diện hiện đại**: Tích hợp Streamlit với biểu đồ trực quan (Waveform, Spectrogram) và trình phát nhạc trực tiếp.

## 🚀 Hướng dẫn cài đặt

### 1. Chuẩn bị môi trường
Yêu cầu: **Python 3.10+** và **Docker Desktop**.

1. **Clone repository:**
   ```bash
   git clone https://github.com/Thang7804/MMDS.git
   cd MMDS
   ```

2. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

### 2. Khởi tạo Cơ sở dữ liệu
Sử dụng Docker để chạy PostgreSQL tích hợp pgvector:
```bash
docker-compose up -d
```
*Lưu ý: Database mặc định chạy ở cổng `5433` để tránh xung đột với Postgres cài sẵn trên máy.*

### 3. Quy trình xử lý dữ liệu (Pipeline)

Thực hiện theo thứ tự sau để thiết lập hệ thống:

1. **Tiền xử lý & Chuẩn hóa:**
   Đặt các file `.wav` gốc vào `raw_audio/` và chạy:
   ```bash
   python normalize_audio.py
   ```
   *Script này sẽ trim silence và cắt lấy đoạn 3 giây chất lượng nhất (Sliding Window).*

2. **Khởi tạo Database:**
   ```bash
   python db/init_postgres.py
   ```

3. **Trích xuất đặc trưng (Lần 1 - RAW):**
   ```bash
   python extract_features.py
   ```
   *(Lúc này hệ thống sẽ báo ⚠️ SCALER NOT FOUND và lưu vector ở dạng thô).*

4. **Tạo bộ chuẩn hóa (Fit Scaler):**
   ```bash
   python fit_scaler.py
   ```
   *Tính toán Mean/Std từ database để tạo file `scaler.pkl`.*

5. **Cập nhật Database (Lần 2 - NORMALIZE):**
   Chạy lại script trích xuất để chuẩn hóa toàn bộ vector trong DB:
   ```bash
   python extract_features.py
   ```
   *(Hệ thống sẽ báo ✅ SCALER LOADED).*

## 🖥️ Cách sử dụng

### Chạy giao diện Web (Streamlit)
```bash
streamlit run app.py
```
Giao diện cho phép upload file, xem phân tích âm thanh và tìm kiếm các giọng nói tương đồng nhất.

### Tìm kiếm qua dòng lệnh (CLI)
```bash
python search_audio.py "đường/dẫn/file.wav" --k 5 --verbose
```

## 📂 Cấu trúc dự án
- `app.py`: Giao diện người dùng Streamlit.
- `extract_features.py`: Trích xuất đặc trưng và đồng bộ database.
- `normalize_audio.py`: Tiền xử lý âm thanh (Trim, Crop, Normalize).
- `fit_scaler.py`: Huấn luyện bộ chuẩn hóa StandardScaler.
- `search_audio.py`: Công cụ tìm kiếm CLI.
- `db/`: Chứa SQL schema và script khởi tạo DB.

## 🛠️ Công nghệ sử dụng
- **Librosa / SoundFile**: Xử lý tín hiệu số âm thanh.
- **Scikit-learn**: Chuẩn hóa dữ liệu (StandardScaler).
- **PostgreSQL + pgvector**: Cơ sở dữ liệu vector.
- **Psycopg 3**: Database driver thế hệ mới.
- **Joblib**: Lưu trữ mô hình scaler.

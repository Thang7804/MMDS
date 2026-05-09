import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import psycopg
from pathlib import Path
import tempfile
import os

from extract_features import extract_features, get_connection_string, validate_query_audio

# No-Model Cluster Search
st.set_page_config(page_title="MMDS", layout="wide")

st.title("🎙️ Multimedia Audio Retrieval System")
st.markdown("### Chế độ: Tìm kiếm theo Cụm (SQL-based Retrieval)")

uploaded_file = st.file_uploader("Chọn file .wav để tìm kiếm", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = Path(tmp_file.name)

    st.success(f"File: {uploaded_file.name}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Audio Preview")
        st.audio(uploaded_file)
    with col2:
        st.subheader(" Waveform")
        y, sr = librosa.load(tmp_path, sr=16000)
        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)

    if st.button("🔍 Start Search"):
        with st.spinner("Đang kiểm tra chất lượng âm thanh..."):
            try:
                y_load, sr_load = librosa.load(tmp_path, sr=16000)
                is_valid, msg = validate_query_audio(y_load)
                if not is_valid:
                    st.warning(f"⚠️ {msg}")
                    st.stop()
                
                st.info("✅ Chất lượng âm thanh đạt yêu cầu.")
                
                # 1. Trích xuất đặc trưng
                f_q = extract_features(tmp_path)
                s_q, std_q = f_q["mfcc_static"], f_q["mfcc_std"]
                
                with psycopg.connect(get_connection_string()) as conn:
                    with conn.cursor() as cur:
                        # BƯỚC 1: Tìm cluster_id của file gần nhất (SQL only)
                        cur.execute("""
                            SELECT cluster_id 
                            FROM vw_audio_feature_vectors 
                            ORDER BY (mfcc_static <=> %s::vector) + (mfcc_std <=> %s::vector)
                            LIMIT 1
                        """, (s_q, std_q))
                        
                        res = cur.fetchone()
                        if not res or res[0] is None:
                            st.error("Chưa có dữ liệu phân cụm. Vui lòng chạy cluster_voices.py trước.")
                            st.stop()
                        
                        cluster_id = res[0]
                        st.info(f"Phân tích: Giọng nói thuộc Cụm số **{cluster_id}** (Xác định bằng SQL)")

                        # BƯỚC 2: Truy vấn trong Cluster
                        query_sql = """
                            SELECT 
                                v.file_name, 
                                v.pitch_mean, v.average_energy, v.zcr_mean, 
                                v.centroid_mean, v.bandwidth_mean, v.harmonicity,
                                (1 - ((v.mfcc_static <=> %s::vector) + (v.mfcc_std <=> %s::vector)) / 2.0) as score,
                                f.file_path
                            FROM vw_audio_feature_vectors v
                            JOIN audio_files f ON f.file_name = v.file_name
                            WHERE v.cluster_id = %s
                            ORDER BY (v.mfcc_static <=> %s::vector) + (v.mfcc_std <=> %s::vector)
                            LIMIT 5
                        """
                        cur.execute(query_sql, (s_q, std_q, cluster_id, s_q, std_q))
                        results = cur.fetchall()

                st.subheader("🏆 Kết quả Tìm kiếm")
                for i, (name, pitch, energy, zcr, cnt, bnd, har, score, path) in enumerate(results, 1):
                    with st.expander(f"Hạng {i}: {name} | Độ tương đồng: {score:.2%}"):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.write(f"**File:** {name}")
                            if os.path.exists(path): st.audio(path)
                        with c2:
                            st.write(f"**Pitch:** {pitch:.1f} Hz")
                            st.write(f"**Energy:** {energy:.4f}")
                        with c3:
                            st.write(f"**ZCR:** {zcr:.4f}")
                            st.write(f"**Harmonicity:** {har:.4f}")
                        st.progress(score)
                                
            except Exception as e:
                st.error(f"Error: {e}")
            finally: 
                if tmp_path.exists(): os.remove(tmp_path)
else:
    st.info("Upload file để bắt đầu.")

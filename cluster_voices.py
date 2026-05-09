"""
cluster_voices.py
========================================
Phan cum K-means su dung StandardScaler (Chuan hoa du lieu ve Mean=0, Std=1).
Day la buoc bat buoc khi dung khoang cach Euclidean (L2) trong K-means.
"""

import numpy as np
import psycopg
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from extract_features import get_connection_string, PROJECT_ROOT

def build_raw_vector(row):
    """Gop dac trung tho, dam bao tat ca deu la mang 1 chieu."""
    # row: [p, e, z, c, b, h, static, std]
    scalars = np.atleast_1d(np.array(row[0:6], dtype=float))
    
    # Xu ly MFCC (Co the bi tra ve duoi dang chuoi vector tu Postgres)
    def parse_vec(v):
        if isinstance(v, str):
            # Bo dau ngoac [] va chuyen sang float array
            return np.fromstring(v.strip('[]'), sep=',')
        return np.atleast_1d(np.array(v, dtype=float))

    mfcc_s = parse_vec(row[6])
    mfcc_std = parse_vec(row[7])
    
    return np.concatenate([scalars, mfcc_s, mfcc_std])

def perform_clustering(n_clusters=15):
    conn_str = get_connection_string()
    
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            print("[INFO] Dang doc dac trung tho tu Database...")
            cur.execute("""
                SELECT 
                    audio_id, pitch_mean, average_energy, zcr_mean,
                    centroid_mean, bandwidth_mean, harmonicity,
                    mfcc_static, mfcc_std
                FROM audio_features
            """)
            rows = cur.fetchall()
            
            if not rows:
                print("[ERROR] Database trong.")
                return

            audio_ids = []
            raw_vectors = []
            
            for r in rows:
                audio_ids.append(r[0])
                raw_vectors.append(build_raw_vector(r[1:]))
            
            raw_vectors = np.array(raw_vectors)
            
            # --- BUOC QUAN TRONG: StandardScaler cho K-means ---
            print("[INFO] Dang ap dung StandardScaler cho Clustering (K-means)...")
            scaler = StandardScaler()
            scaled_vectors = scaler.fit_transform(raw_vectors)
            
            print(f"[INFO] Dang phan cum {len(scaled_vectors)} file thanh {n_clusters} nhom...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled_vectors)
            
            print("[INFO] Dang luu ket qua Cluster ID...")
            for aid, label in zip(audio_ids, labels):
                cur.execute("UPDATE audio_features SET cluster_id = %s WHERE audio_id = %s", (int(label), aid))
            
            conn.commit()
            print(f"[SUCCESS] Da phan cum xong .")

if __name__ == "__main__":
    perform_clustering()


import numpy as np
import psycopg
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from extract_features import get_connection_string, SCALER_PATH
from tqdm import tqdm

def parse_pg_vector(val):
    """Chuyen doi gia tri tu Postgres Vector (chuoi hoac list) sang list floats"""
    if val is None:
        return []
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    if isinstance(val, str):
        # Xu ly chuoi dang [1.2, 3.4, ...]
        return [float(x) for x in val.strip("[]").split(",")]
    return []

def fit_scaler_from_database():
    print("[INFO] Loading RAW features from database to fit scaler...")
    
    conn_str = get_connection_string()
    all_rows = []
    
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            # Lay 7 scalars thô + 13 MFCC Static + 13 MFCC Std
            cur.execute("""
                SELECT 
                    pitch_mean, zcr_mean, voiced_ratio, average_energy, 
                    harmonicity, centroid_mean, bandwidth_mean,
                    mfcc_static, mfcc_std
                FROM audio_features
            """)
            
            rows = cur.fetchall()
            if not rows:
                print("[ERROR] Không tìm thấy dữ liệu đặc trưng nào trong database.")
                return False
            
            for r in tqdm(rows, desc="Processing rows"):
                try:
                    # Lay 7 scalars thô
                    scalars = [float(x) if x is not None else 0.0 for x in r[:7]]
                    
                    # Parse 2 vectors MFCC
                    mfcc_static = parse_pg_vector(r[7])
                    mfcc_std = parse_pg_vector(r[8])
                    
                    # Ghep thanh vector 33D
                    full_vector = scalars + mfcc_static + mfcc_std
                    
                    if len(full_vector) == 33:
                        all_rows.append(full_vector)
                    else:
                        print(f"\n[SKIP] Vector không đủ 33 chiều (đang có {len(full_vector)})")
                except Exception as e:
                    print(f"\n[ERROR] Lỗi khi xử lý dòng: {e}")
                    continue

    if not all_rows:
        print("[ERROR] Không có dữ liệu hợp lệ để fit scaler.")
        return False

    X = np.array(all_rows, dtype=np.float32)
    print(f"[INFO] Fitting scaler on matrix {X.shape}")
    
    scaler = StandardScaler()
    scaler.fit(X)
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"[SUCCESS] Scaler saved to {SCALER_PATH}")
    return True

if __name__ == "__main__":
    fit_scaler_from_database()

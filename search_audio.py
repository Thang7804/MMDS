import argparse
from pathlib import Path
import psycopg
from extract_features import extract_features, get_connection_string

def search_top_k(query_audio_path: str, top_k: int = 5):
    path = Path(query_audio_path)
    if not path.exists(): return print(f"[ERROR] Khong tim thay: {query_audio_path}")
    print(f"[INFO] Analyzing query: {path.name}")
    
    # 1. Trich xuat dac trung query
    f_q = extract_features(path)
    s_q, std_q = f_q["mfcc_static"], f_q["mfcc_std"]
    
    with psycopg.connect(get_connection_string()) as conn:
        with conn.cursor() as cur:
            # BUOC 1: Tim cluster cua file tuong dong nhat (Global Search Top-1)
            # Khong dung model, chi dung SQL de xac dinh cluster dai dien
            cur.execute("""
                SELECT cluster_id 
                FROM vw_audio_feature_vectors 
                ORDER BY (mfcc_static <=> %s::vector) + (mfcc_std <=> %s::vector)
                LIMIT 1
            """, (s_q, std_q))
            
            res = cur.fetchone()
            if not res or res[0] is None:
                return print("[ERROR] Khong the xac dinh cluster. Hay chay cluster_voices.py truoc.")
            
            cluster_id = res[0]
            print(f"[INFO] Mượn Cluster ID từ file tương đồng nhất: {cluster_id}")
            
            # BUOC 2: Tim kiem Top-K trong cluster vua tim duoc
            query_sql = """
                SELECT 
                    v.file_name, 
                    v.pitch_mean, v.average_energy, v.zcr_mean, 
                    (1 - ((v.mfcc_static <=> %s::vector) + (v.mfcc_std <=> %s::vector)) / 2.0) as score
                FROM vw_audio_feature_vectors v
                WHERE v.cluster_id = %s
                ORDER BY (v.mfcc_static <=> %s::vector) + (v.mfcc_std <=> %s::vector)
                LIMIT %s
            """
            cur.execute(query_sql, (s_q, std_q, cluster_id, s_q, std_q, top_k))
            results = cur.fetchall()

            print("\n" + "="*85)
            print(f"{'Hạng':<5} | {'File Name':<30} | {'Pitch':<8} | {'Energy':<8} | {'Sim Score'}")
            print("-" * 85)
            for i, (name, pitch, energy, zcr, score) in enumerate(results, 1):
                print(f"{i:<5} | {str(name)[:30]:<30} | {pitch:>6.1f} | {energy:>8.4f} | {score:>9.2%}")
            print("="*85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    search_top_k(args.query, args.k)

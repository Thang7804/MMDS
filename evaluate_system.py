import psycopg
import random
from extract_features import get_connection_string

def evaluate(n_tests=50):
    conn_str = get_connection_string()
    
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            # Lấy thông tin file
            cur.execute("""
                SELECT f.id, f.speaker_label, 
                       af.mfcc_static, af.mfcc_std
                FROM audio_files f
                JOIN audio_features af ON f.id = af.audio_id
            """)
            all_data = cur.fetchall()
            
            if not all_data:
                print("[ERROR] Database trong.")
                return

            test_samples = random.sample(all_data, min(n_tests, len(all_data)))
            print(f"[INFO] Evaluating Performance (SQL-based Cluster Search)...")
            
            hits = 0
            for row in test_samples:
                audio_id, label = row[0], row[1]
                s_q, std_q = row[2], row[3]
                
                # 1. Tìm cluster của file gần nhất (SQL only)
                cur.execute("""
                    SELECT cluster_id 
                    FROM audio_features 
                    WHERE audio_id != %s
                    ORDER BY (mfcc_static <=> %s::vector) + (mfcc_std <=> %s::vector)
                    LIMIT 1
                """, (audio_id, s_q, std_q))
                
                res = cur.fetchone()
                if not res or res[0] is None: continue
                cluster_id = res[0]
                
                # 2. Tìm kiếm trong Cluster
                cur.execute("""
                    SELECT f.speaker_label
                    FROM audio_features af
                    JOIN audio_files f ON f.id = af.audio_id
                    WHERE af.cluster_id = %s AND f.id != %s
                    ORDER BY (af.mfcc_static <=> %s::vector) + (af.mfcc_std <=> %s::vector)
                    LIMIT 1
                """, (cluster_id, audio_id, s_q, std_q))
                
                result = cur.fetchone()
                if result and result[0] == label:
                    hits += 1
            
            accuracy = (hits / len(test_samples)) * 100
            print("\n" + "="*45)
            print("KẾT QUẢ ĐÁNH GIÁ")
            print("="*45)
            print(f"Tổng số mẫu thử : {len(test_samples)}")
            print(f"Độ chính xác    : {accuracy:.2f}%")
            print("="*45)

if __name__ == "__main__":
    evaluate()

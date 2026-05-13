

from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import joblib
import librosa
import numpy as np
import psycopg
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "normalized_audio"
DB_DIR = PROJECT_ROOT / "db"
ENV_PATH = DB_DIR / ".env.postgres"
SCALER_PATH = PROJECT_ROOT / "scaler.pkl"

SAMPLE_RATE = 16000
FRAME_LENGTH = 2048
HOP_LENGTH = 512

def load_env_file(env_path: Path) -> None:
    if not env_path.exists(): return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")

def get_connection_string() -> str:
    load_env_file(ENV_PATH)
    return f"host={os.getenv('PGHOST', 'localhost')} port={os.getenv('PGPORT', '5432')} dbname={os.getenv('PGDATABASE', 'male_voice_db')} user={os.getenv('PGUSER', 'postgres')} password={os.getenv('PGPASSWORD')}"

def safe_mean(v: np.ndarray) -> float:
    return float(np.mean(v)) if v.size else 0.0


def compute_voiced_ratio(y: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Ti le khung duoc pyin danh dau la voiced (tieng noi co chu ky co ban).
    Cung tham so voi validate_query_audio de thong nhat.
    """
    # librosa.pyin returns (f0, voiced_flag, voiced_prob)
    _, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
    if len(voiced_flag) == 0:
        return 0.0
    return float(np.sum(voiced_flag) / len(voiced_flag))


def validate_query_audio(y: np.ndarray) -> tuple[bool, str]:
    """
    Kiem tra file query truoc khi tim kiem:
    1. voiced_ratio < 0.1 -> Khong phai tieng nguoi (ha xuong 10% de linh hoat hon)
    2. pitch_mean > 260 -> Co the la giong nu (fmax=300, ngưỡng 260Hz de tranh false positive cho giong nam)
    """
    # Tu dong trim silence (dung top_db=25 de cat manh hon, dong bo voi normalize)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    
    if len(y_trimmed) == 0:
        return False, "Âm thanh rỗng hoặc chỉ có khoảng lặng."

    # Dung pyin de lay ca pitch va voiced_flag trong 1 lan goi
    # librosa.pyin returns (f0, voiced_flag, voiced_prob)
    f0, voiced_flag, _ = librosa.pyin(y_trimmed, fmin=50, fmax=300, sr=SAMPLE_RATE)
    
    # 1. Kiem tra voiced_ratio
    voiced_ratio = float(np.sum(voiced_flag) / len(voiced_flag)) if len(voiced_flag) > 0 else 0.0
    
    if voiced_ratio < 0.1:
        return False, f"Không phát hiện giọng người (Voiced Ratio: {voiced_ratio:.1%}). Yêu cầu > 10%."
    
    # 2. Kiem tra gioi tinh (Giong nam thuong < 260Hz)
    pitch_mean = float(np.nanmean(f0)) if np.any(~np.isnan(f0)) else 0.0
    
    if pitch_mean > 260:
        return False, f"Phát hiện giọng nữ hoặc tông cao ({pitch_mean:.0f}Hz). Hệ thống chỉ hỗ trợ giọng nam."
    
    return True, "OK"


def build_combined_33d_vector(f: dict) -> list:
    """
    Build 33D combined feature vector với thứ tự cố định:
    [pitch, zcr, vr, energy, harm, centr, band] + 13 MFCC + 13 MFCC_std
    """
    raw_scalars = [
        f["pitch_mean"],
        f["zcr_mean"],
        f["voiced_ratio"],
        f["average_energy"],
        f["harmonicity"],
        f["centroid_mean"],
        f["bandwidth_mean"]
    ]
    
    # Tổng cộng: 7 scalars + 13 MFCC + 13 MFCC_std = 33D
    # Ep kieu ve float de tranh loi mixed types (float/int) khi thuc hien insert postgres
    return [float(x) for x in (raw_scalars + f["mfcc_static"] + f["mfcc_std"])]


def load_scaler() -> StandardScaler | None:
    if not SCALER_PATH.exists():
        return None
    return joblib.load(SCALER_PATH)


def transform_combined_vector(vector: list[float], scaler: StandardScaler | None) -> list[float]:
    if scaler is None:
        return vector
    # Chuyen doi vector ve dang chuan hoa
    return list(scaler.transform([vector])[0])


def extract_features(audio_path: Path) -> dict[str, Any]:
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    
    # Tự động cắt khoảng lặng (VAD) để tránh làm loãng đặc trưng (nhất là khi query rất ngắn)
    y, _ = librosa.effects.trim(y, top_db=25)
    
    # Đảm bảo y không rỗng sau khi trim
    if len(y) == 0:
        y = np.zeros(int(0.5 * SAMPLE_RATE)) # Dummy 0.5s silence nếu không có tiếng

    file_info = sf.info(audio_path)

    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    avg_energy = safe_mean(rms)
    
    # Tinh Zero Crossing Rate

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    zcr_mean = safe_mean(zcr)
    
    # Tinh Pitch mean (Dung pyin de dong bo voi validate)
    # librosa.pyin returns (f0, voiced_flag, voiced_prob)
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
    pitch_mean = float(np.nanmean(f0)) if np.any(~np.isnan(f0)) else 0.0
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    centroid_mean = safe_mean(centroid)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    bandwidth_mean = safe_mean(bandwidth)
    harmonicity = float(np.sum(librosa.effects.harmonic(y)**2) / np.sum(y**2)) if np.sum(y**2) > 0 else 0.0

    voiced_ratio = compute_voiced_ratio(y, sr)

    mfcc_s = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    
    feats = {
        "average_energy": avg_energy, "zcr_mean": zcr_mean,
        "pitch_mean": pitch_mean, "centroid_mean": centroid_mean, "bandwidth_mean": bandwidth_mean, "harmonicity": harmonicity,
        "voiced_ratio": voiced_ratio,
        "mfcc_static": [safe_mean(m) for m in mfcc_s], "mfcc_std": [float(np.std(m)) for m in mfcc_s]
    }
    
    feats.update({
        "file_name": audio_path.name, "file_path": str(audio_path.resolve()),
        "sample_rate": sr, "duration_sec": float(librosa.get_duration(y=y, sr=sr)),
        "bit_depth": int("".join(filter(str.isdigit, file_info.subtype_info)) or 16)
    })
    return feats

def main() -> None:
    audio_files = sorted(list(INPUT_DIR.rglob("*.wav")))
    conn_str = get_connection_string()
    scaler = load_scaler()  # Load scaler once, not in every iteration
    
    if scaler is not None:
        print(f"✅ [SCALER LOADED] Tim thay bo chuan hoa tai {SCALER_PATH}. Vector se duoc CHUAN HOA.")
    else:
        print(f"⚠️ [SCALER NOT FOUND] Khong tim thay {SCALER_PATH}. Vector se o dang THO (RAW).")
        print("   (Hay chay fit_scaler.py sau khi da index raw features de tao bo chuan hoa)")

    with psycopg.connect(conn_str) as conn:
        for audio_path in tqdm(audio_files, desc="Extracting features"):
            try:
                with conn.transaction():
                    with conn.cursor() as cur:
                        f = extract_features(audio_path)
                        # Build combined 33D vector
                        combined_vector = build_combined_33d_vector(f)

                        # If a scaler is already available, transform vectors before saving
                        if scaler is not None:
                            combined_vector = transform_combined_vector(combined_vector, scaler)
                        
                        cur.execute("""
                            INSERT INTO audio_files (file_name, file_path, sample_rate, duration_sec, bit_depth) 
                            VALUES (%s,%s,%s,%s,%s) 
                            ON CONFLICT (file_path) 
                            DO UPDATE SET file_name = EXCLUDED.file_name 
                            RETURNING id
                        """, (f['file_name'], f['file_path'], f['sample_rate'], f['duration_sec'], f['bit_depth']))
                        aid = cur.fetchone()[0]
                        cur.execute("""
                            INSERT INTO audio_features (audio_id, average_energy, zcr_mean, pitch_mean, centroid_mean, bandwidth_mean, harmonicity, voiced_ratio, mfcc_static, mfcc_std, combined_features_vector)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (audio_id) DO UPDATE SET
                                average_energy = EXCLUDED.average_energy,
                                zcr_mean = EXCLUDED.zcr_mean,
                                pitch_mean = EXCLUDED.pitch_mean,
                                centroid_mean = EXCLUDED.centroid_mean,
                                bandwidth_mean = EXCLUDED.bandwidth_mean,
                                harmonicity = EXCLUDED.harmonicity,
                                voiced_ratio = EXCLUDED.voiced_ratio,
                                mfcc_static = EXCLUDED.mfcc_static,
                                mfcc_std = EXCLUDED.mfcc_std,
                                combined_features_vector = EXCLUDED.combined_features_vector
                        """, (aid, f['average_energy'], f['zcr_mean'], f['pitch_mean'], f['centroid_mean'], f['bandwidth_mean'], f['harmonicity'], f['voiced_ratio'], f['mfcc_static'], f['mfcc_std'], combined_vector))
            except Exception as exc: print(f"\n[ERROR] {audio_path.name}: {exc}")

if __name__ == "__main__": main()

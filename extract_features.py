

from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import librosa
import numpy as np
import psycopg
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "normalized_audio"
DB_DIR = PROJECT_ROOT / "db"
ENV_PATH = DB_DIR / ".env.postgres"

SAMPLE_RATE = 16000
FRAME_LENGTH = 2048
HOP_LENGTH = 512
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

def validate_query_audio(y: np.ndarray) -> tuple[bool, str]:
    """
    Kiem tra file query truoc khi tim kiem:
    1. voiced_ratio < 0.2 -> Khong phai tieng nguoi
    2. pitch_mean > 200 -> Co the la giong nu
    """
    # 1. Chinh xac: Kiem tra voiced_ratio bang pyin
    f0, _, voiced_flag = librosa.pyin(y, fmin=50, fmax=400, sr=SAMPLE_RATE)
    voiced_ratio = np.sum(voiced_flag) / len(voiced_flag) if len(voiced_flag) > 0 else 0
    if voiced_ratio < 0.2:
        return False, "Không phát hiện giọng người (Voiced Ratio < 20%)."
    
    # 2. Kiem tra gioi tinh (Giong nam < 200Hz)
    pitch_mean = float(np.nanmean(f0)) if np.any(~np.isnan(f0)) else 0
    if pitch_mean > 200:
        return False, f"Phát hiện giọng nữ hoặc tông cao ({pitch_mean:.0f}Hz). Hệ thống chỉ hỗ trợ giọng nam."
    
    return True, "OK"


def build_32d_vector(f: dict) -> np.ndarray:
    """Gop dac trung thanh vector 32-chieu (2 Time + 4 Freq + 13 MFCC Mean + 13 MFCC Std)."""
    scalars = [
        f["average_energy"], f["zcr_mean"], 
        f["pitch_mean"], f["centroid_mean"], f["bandwidth_mean"], f["harmonicity"]
    ]
    return np.concatenate([scalars, f["mfcc_static"], f["mfcc_std"]])

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
    
    f0 = librosa.yin(y=y, sr=sr, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C5"))
    pitch_mean = safe_mean(f0[~np.isnan(f0)])
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    centroid_mean = safe_mean(centroid)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    bandwidth_mean = safe_mean(bandwidth)
    harmonicity = float(np.sum(librosa.effects.harmonic(y)**2) / np.sum(y**2)) if np.sum(y**2) > 0 else 0.0

    mfcc_s = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    
    feats = {
        "average_energy": avg_energy, "zcr_mean": zcr_mean,
        "pitch_mean": pitch_mean, "centroid_mean": centroid_mean, "bandwidth_mean": bandwidth_mean, "harmonicity": harmonicity,
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
    with psycopg.connect(conn_str) as conn:
        for audio_path in tqdm(audio_files, desc="Extracting features"):
            try:
                with conn.transaction():
                    with conn.cursor() as cur:
                        f = extract_features(audio_path)
                        cur.execute("INSERT INTO audio_files (file_name, file_path, sample_rate, duration_sec, bit_depth) VALUES (%s,%s,%s,%s,%s) ON CONFLICT (file_path) DO NOTHING RETURNING id", (f['file_name'], f['file_path'], f['sample_rate'], f['duration_sec'], f['bit_depth']))
                        aid = cur.fetchone()[0]
                        cur.execute("""
                            INSERT INTO audio_features (audio_id, average_energy, zcr_mean, pitch_mean, centroid_mean, bandwidth_mean, harmonicity, mfcc_static, mfcc_std)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (audio_id) DO UPDATE SET mfcc_static = EXCLUDED.mfcc_static, mfcc_std = EXCLUDED.mfcc_std
                        """, (aid, f['average_energy'], f['zcr_mean'], f['pitch_mean'], f['centroid_mean'], f['bandwidth_mean'], f['harmonicity'], f['mfcc_static'], f['mfcc_std']))
            except Exception as exc: print(f"\n[ERROR] {audio_path.name}: {exc}")

if __name__ == "__main__": main()

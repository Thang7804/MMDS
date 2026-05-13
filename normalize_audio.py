"""
normalize_audio.py
==================
Chuan hoa toan bo file WAV trong `raw_audio` ve cung dinh dang:
- mono, 16 kHz, 3.0 giay (same duration)
- Trim silence (cat khoang lang dau/cuoi) truoc khi xu ly
- File >= 3s sau trim: cat doan giua -> giu lai
- File < 3s sau trim: bo qua (khong pad, dam bao 100% tieng noi thuc)

Chay:
    pip install librosa soundfile tqdm numpy
    python normalize_audio.py
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Cau hinh
INPUT_DIR = Path("raw_audio")
OUTPUT_DIR = Path("normalized_audio")
TARGET_DURATION = 3.0
SAMPLE_RATE = 16000
AUDIO_EXTENSIONS = {".wav"}

TARGET_SAMPLES = int(TARGET_DURATION * SAMPLE_RATE)
MIN_SPEECH_DURATION = 3.0  # giay - chi giu file >= 3s sau trim (0% silence padding)
MIN_SPEECH_SAMPLES = TARGET_SAMPLES

# Nguong trim silence (dB) - cang cao cang "manh tay" cat
TRIM_TOP_DB = 25


def collect_audio_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def normalize_waveform(y: np.ndarray) -> tuple[np.ndarray | None, str]:
    """
    Chuan hoa waveform ve dung 3.0 giay:
    1. Trim silence dau va cuoi
    2. Kiem tra do dai
    3. Tim cua so 3s co voiced_ratio (tieng noi) cao nhat bang Sliding Window
    """
    # Buoc 1: Trim silence dau va cuoi
    y_trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)

    # Buoc 2: Bo qua file < 3s sau trim
    if len(y_trimmed) < TARGET_SAMPLES:
        return None, "skipped_too_short"
    
    # Buoc 3: Phan tich voiced_flag tren toan bo file (Chay pyin 1 lan duy nhat)
    hop_length = 512
    # librosa.pyin returns (f0, voiced_flag, voiced_prob)
    f0, voiced_flag, _ = librosa.pyin(y_trimmed, fmin=50, fmax=300, sr=SAMPLE_RATE, hop_length=hop_length)
    
    # Buoc 4: Sliding Window tim doan 3s tot nhat
    frames_in_3s = int(TARGET_SAMPLES / hop_length)
    best_voiced_count = -1
    best_start_sample = 0
    
    # Buoc nhay 0.05s de dam bao do chinh xac cao
    step = int(0.05 * SAMPLE_RATE / hop_length) or 1
    
    for i in range(0, len(voiced_flag) - frames_in_3s, step):
        current_voiced_count = np.sum(voiced_flag[i : i + frames_in_3s])
        if current_voiced_count > best_voiced_count:
            best_voiced_count = current_voiced_count
            best_start_sample = i * hop_length

    # Buoc 5: Lay doan 3s tot nhat
    y_3s = y_trimmed[best_start_sample : best_start_sample + TARGET_SAMPLES]
    
    # Buoc 6: Filter cuoi cung dua tren voiced_ratio cua doan duoc chon
    final_vr = best_voiced_count / frames_in_3s
    if final_vr < 0.2:
        return None, "skipped_not_voiced"
    
    return y_3s, "trimmed"


def normalize_audio_file(input_path: Path, output_path: Path) -> str:
    try:
        y, _ = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        y_out, status = normalize_waveform(y)

        if y_out is None:
            return status  # "skipped_too_short"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, y_out, SAMPLE_RATE, subtype="PCM_16")
        return status
    except Exception as exc:
        print(f"[ERROR] {input_path}: {exc}")
        return "error"


def print_summary(stats: dict[str, int], total: int) -> None:
    print("\n" + "=" * 52)
    print("KET QUA CHUAN HOA DATASET (3.0s Fixed Duration)")
    print("=" * 52)
    print(f"Tong file xu ly       : {total}")
    print(f"Giữ lại & Cắt (>=3s)  : {stats['trimmed']} ({stats['trimmed'] / total * 100:.1f}%)")
    print(f"Bỏ qua (<3s)          : {stats['skipped_too_short']} ({stats['skipped_too_short'] / total * 100:.1f}%)")
    print(f"Bỏ qua (Không tiếng)  : {stats.get('skipped_not_voiced', 0)}")
    print(f"Loi                   : {stats['error']}")
    print("=" * 52)
    valid = stats['trimmed']
    print(f"Tong file hop le      : {valid} (Tat ca deu dung 3.0s)")
    print(f"Thu muc output        : {OUTPUT_DIR.resolve()}")


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"[ERROR] Khong tim thay thu muc input: {INPUT_DIR.resolve()}")
        return

    wav_files = collect_audio_files(INPUT_DIR)
    if not wav_files:
        print(f"[ERROR] Khong tim thay file WAV nao trong '{INPUT_DIR}'")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Tim thay {len(wav_files)} file WAV")
    print(f"[INFO] Chuan hoa ve: mono, {SAMPLE_RATE} Hz, {TARGET_DURATION}s")
    print(f"[INFO] Trim silence (top_db={TRIM_TOP_DB}), chi giu file >= 3s")
    print(f"[INFO] Bo qua (skip) file < 3s sau khi trim")
    print(f"[INFO] Output: {OUTPUT_DIR.resolve()}\n")

    stats = {"trimmed": 0, "skipped_too_short": 0, "skipped_not_voiced": 0, "error": 0}

    for input_path in tqdm(wav_files, desc="Dang chuan hoa", unit="file"):
        relative_path = input_path.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / relative_path
        status = normalize_audio_file(input_path, output_path)
        stats[status] += 1

    print_summary(stats, len(wav_files))


if __name__ == "__main__":
    main()

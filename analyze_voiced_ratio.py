"""
analyze_voiced_ratio.py
======================
Script nay dung de phan tich mat do "voiced" (tieng noi thuc) trong dataset.
Ket qua giup ban chon nguong (threshold) phu hop de loai bo file rac.
"""

import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Cau hinh
INPUT_DIR = Path("raw_audio")
SAMPLE_RATE = 16000

def analyze():
    audio_files = sorted(list(INPUT_DIR.rglob("*.wav")))
    if not audio_files:
        print(f"[ERROR] Khong tim thay file WAV trong {INPUT_DIR.resolve()}")
        return

    print(f"[INFO] Dang phan tich {len(audio_files)} file. Vui long doi (pyin kha cham)...")
    
    voiced_ratios = []

    # Gioi han 100 file de test nhanh neu dataset qua lon (Ban co the bo [:100] de chay het)
    for file_path in tqdm(audio_files, desc="Processing"):
        try:
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            # Cat bo im lang dau/cuoi truoc khi tinh voiced_ratio
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            
            if len(y_trimmed) == 0:
                voiced_ratios.append(0.0)
                continue

            # Dung pyin de xac dinh khung hinh nao la tieng nguoi
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y_trimmed, fmin=50, fmax=400, sr=SAMPLE_RATE
            )
            
            v_ratio = np.sum(voiced_flag) / len(voiced_flag) if len(voiced_flag) > 0 else 0
            voiced_ratios.append(v_ratio)
            
        except Exception as e:
            print(f"\n[ERROR] {file_path.name}: {e}")

    voiced_ratios = np.array(voiced_ratios)

    print("\n" + "="*45)
    print("KET QUA PHAN TICH VOICED RATIO")
    print("="*45)
    print(f"Tong so file  : {len(voiced_ratios)}")
    print(f"Min           : {np.min(voiced_ratios):.4f}")
    print(f"Max           : {np.max(voiced_ratios):.4f}")
    print(f"Mean (Trung binh): {np.mean(voiced_ratios):.4f}")
    print(f"Median (Trung vi): {np.median(voiced_ratios):.4f}")
    print(f"Std (Do lech) : {np.std(voiced_ratios):.4f}")
    print("-" * 45)

    # Cong thuc goi y: Mean - 2*Std
    suggested_threshold = np.mean(voiced_ratios) - 2 * np.std(voiced_ratios)
    print(f"NGUONG GOI Y (Mean - 2*Std): {max(0, suggested_threshold):.4f}")
    print("="*45)
    
    # Ve bieu do phan bo de truc quan hoa
    plt.figure(figsize=(10, 6))
    plt.hist(voiced_ratios, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(suggested_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({suggested_threshold:.2f})')
    plt.title("Phân bố Voiced Ratio trong Dataset")
    plt.xlabel("Voiced Ratio (0.0 - 1.0)")
    plt.ylabel("Số lượng file")
    plt.legend()
    plt.savefig("voiced_ratio_distribution.png")
    print("[SUCCESS] Da luu bieu do phan bo vao file 'voiced_ratio_distribution.png'")

if __name__ == "__main__":
    analyze()

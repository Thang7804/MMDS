"""
🎙️ MMDS - Audio Search using Combined 33D Feature Vector
=========================================================

This version uses a single combined feature vector containing:
- 7 normalized scalar features (pitch, zcr, voiced_ratio, energy, harmonicity, centroid, bandwidth)
- 13 MFCC static values
- 13 MFCC std values

Total: 33-dimensional vector for efficient cosine similarity search

Vector is standardized using StandardScaler fitted on database vectors.
"""

import argparse
from pathlib import Path
import psycopg
import librosa
import numpy as np
from extract_features import (
    extract_features, 
    get_connection_string, 
    validate_query_audio,
    build_combined_33d_vector,
    transform_combined_vector,
    load_scaler,
    SAMPLE_RATE
)


def search_top_k(query_audio_path: str, top_k: int = 5, verbose: bool = False):
    """
    Search database for similar audio using combined 33D feature vector.
    
    Args:
        query_audio_path: Path to query audio file
        top_k: Number of top results to return (default: 5)
        verbose: Print detailed feature information
    """
    path = Path(query_audio_path)
    if not path.exists():
        print(f"[ERROR] File not found: {query_audio_path}")
        return
    
    print(f"[INFO] Analyzing query: {path.name}")
    
    # Load scaler
    scaler = load_scaler()
    if scaler is None:
        print("[WARNING] Scaler not found. Run fit_scaler.py first to standardize vectors.")
        print("[WARNING] Proceeding without scaling...")
    
    # 1. Validate query audio
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    is_valid, msg = validate_query_audio(y)
    if not is_valid:
        print(f"[ERROR] {msg}")
        return
    
    # 2. Extract features and build combined vector
    f_q = extract_features(path)
    query_combined_vector = build_combined_33d_vector(f_q)
    
    # 3. Transform with scaler if available
    query_combined_vector = transform_combined_vector(query_combined_vector, scaler)
    
    # 4. Search database
    with psycopg.connect(get_connection_string()) as conn:
        with conn.cursor() as cur:
            # Query using combined vector (cosine similarity)
            cur.execute("""
                SELECT
                    f.file_name,
                    af.pitch_mean, 
                    af.average_energy, 
                    af.zcr_mean, 
                    af.voiced_ratio,
                    af.harmonicity,
                    (1.0 - (af.combined_features_vector <=> %s::vector)) AS similarity_score,
                    f.file_path
                FROM audio_features af
                JOIN audio_files f ON f.id = af.audio_id
                ORDER BY similarity_score DESC
                LIMIT %s
            """, (query_combined_vector, top_k))
            
            results = cur.fetchall()
    
    if not results:
        print("[ERROR] No results. Check if extract_features.py has been run and data exists in DB.")
        return
    
    # 5. Display results
    print("\n" + "="*120)
    print(f"{'Rank':<5} | {'File Name':<28} | {'Pitch':<8} | {'Energy':<8} | {'ZCR':<7} | {'Voiced%':<8} | {'Sim%':<8}")
    print("-"*120)
    
    for i, (name, pitch, energy, zcr, vr, harmonicity, score, path) in enumerate(results, 1):
        print(f"{i:<5} | {str(name)[:28]:<28} | {pitch:>7.1f} | {energy:>7.4f} | {zcr:>6.4f} | {vr:>7.1%} | {score:>7.1%}")
    
    print("="*120)
    
    if verbose:
        print("\n[QUERY AUDIO FEATURES - RAW VALUES]")
        print(f"  Pitch: {f_q['pitch_mean']:.1f} Hz")
        print(f"  ZCR: {f_q['zcr_mean']:.4f}")
        print(f"  Voiced Ratio: {f_q['voiced_ratio']:.1%}")
        print(f"  Energy: {f_q['average_energy']:.4f}")
        print(f"  Harmonicity: {f_q['harmonicity']:.4f}")
        print(f"  Centroid: {f_q['centroid_mean']:.1f} Hz")
        print(f"  Bandwidth: {f_q['bandwidth_mean']:.1f} Hz")
        print(f"\n[COMBINED VECTOR AFTER STANDARDSCALER] (33D, z-score normalized)")
        print(f"  Scaler status: {'Active' if scaler else 'Not loaded'}")
        print(f"  7 Raw Scalars (scaled): {[f'{v:.4f}' for v in query_combined_vector[:7]]}")
        print(f"  13 MFCC Static (scaled): {[f'{v:.4f}' for v in query_combined_vector[7:20]]}")
        print(f"  13 MFCC Std (scaled): {[f'{v:.4f}' for v in query_combined_vector[20:33]]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search audio database using combined 33D feature vector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python search_audio.py "normalized_audio/speaker1/audio.wav"
  python search_audio.py "normalized_audio/speaker1/audio.wav" --k 10 --verbose
        """
    )
    parser.add_argument("query", help="Path to query audio file (.wav)")
    parser.add_argument("--k", type=int, default=5, help="Number of top results (default: 5)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed feature information")
    
    args = parser.parse_args()
    search_top_k(args.query, args.k, args.verbose)

"""
🎙️ MMDS - Male Voice Retrieval System
No Sidebar Version
"""

import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import psycopg
from pathlib import Path
import tempfile
import os

from extract_features import (
    extract_features,
    get_connection_string,
    validate_query_audio,
    build_combined_33d_vector,
    load_scaler,
    transform_combined_vector,
    SAMPLE_RATE
)

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="🎙️ MMDS - Audio Retrieval",
    layout="wide"
)

# ============================================================================
# Header
# ============================================================================

st.title("🎙️ Male Voice Retrieval System")


# ============================================================================
# Configuration
# ============================================================================

top_k = 5
show_waveform = True
show_spectrogram = True
show_features = True
show_vector = False

# ============================================================================
# Upload Section
# ============================================================================

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📁 Upload Query Audio (.wav)",
        type=["wav"]
    )

with col2:
    st.write("")
    st.write("")

    run_search = st.button(
        "🔍 Search",
        use_container_width=True,
        type="primary",
        key="search_button"
    )

# ============================================================================
# Main Processing
# ============================================================================

if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = Path(tmp_file.name)

    st.success(f"✅ Loaded: **{uploaded_file.name}**")

    # Load audio once
    y, sr = librosa.load(
        tmp_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # =========================================================================
    # Tabs
    # =========================================================================

    tab_preview, tab_results = st.tabs([
        "📊 Audio Preview",
        "🏆 Search Results"
    ])

    # =========================================================================
    # PREVIEW TAB
    # =========================================================================

    with tab_preview:

        # ---------------------------------------------------------------------
        # Waveform
        # ---------------------------------------------------------------------

        if show_waveform:

            st.subheader("🌊 Waveform")

            fig, ax = plt.subplots(figsize=(12, 3))

            librosa.display.waveshow(
                y,
                sr=sr,
                ax=ax,
                color="#1f77b4"
            )

            ax.set_title(f"Query Audio: {uploaded_file.name}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")

            st.pyplot(fig, use_container_width=True)

        # ---------------------------------------------------------------------
        # Spectrogram
        # ---------------------------------------------------------------------

        if show_spectrogram:

            st.subheader("📈 Spectrogram")

            D = librosa.stft(y)

            S_db = librosa.power_to_db(
                np.abs(D) ** 2,
                ref=np.max
            )

            fig, ax = plt.subplots(figsize=(12, 4))

            img = librosa.display.specshow(
                S_db,
                sr=sr,
                x_axis="time",
                y_axis="log",
                ax=ax,
                cmap="viridis"
            )

            ax.set_title("Spectrogram")

            fig.colorbar(
                img,
                ax=ax,
                format="%+2.0f dB",
                label="Power (dB)"
            )

            st.pyplot(fig, use_container_width=True)

        # ---------------------------------------------------------------------
        # Audio Player
        # ---------------------------------------------------------------------

        st.subheader("🔊 Audio Playback")
        st.audio(uploaded_file)

    # =========================================================================
    # SEARCH TAB
    # =========================================================================

    with tab_results:

        if run_search:

            # -----------------------------------------------------------------
            # Validate Audio
            # -----------------------------------------------------------------

            with st.spinner("🔄 Validating audio..."):

                is_valid, msg = validate_query_audio(y)

                if not is_valid:
                    st.error(f"❌ Validation Failed: {msg}")
                    st.stop()

                st.success("✅ Audio validation passed")

            # -----------------------------------------------------------------
            # Extract Features
            # -----------------------------------------------------------------

            with st.spinner("📊 Extracting features..."):

                f_q = extract_features(tmp_path)

                combined_vector = build_combined_33d_vector(f_q)
                
                # BUG FIX: Phải scale query vector trước khi tìm kiếm
                scaler = load_scaler()
                if scaler is not None:
                    combined_vector = transform_combined_vector(combined_vector, scaler)

            # -----------------------------------------------------------------
            # Query Features
            # -----------------------------------------------------------------

            if show_features:

                st.subheader("📋 Query Audio Features")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "🎵 Pitch",
                        f"{f_q['pitch_mean']:.0f} Hz"
                    )

                with col2:
                    st.metric(
                        "📊 Energy",
                        f"{f_q['average_energy']:.4f}"
                    )

                with col3:
                    st.metric(
                        "📈 ZCR",
                        f"{f_q['zcr_mean']:.4f}"
                    )

                with col4:
                    st.metric(
                        "🗣️ Voiced%",
                        f"{f_q['voiced_ratio']:.1%}"
                    )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "🎼 Harmonicity",
                        f"{f_q['harmonicity']:.4f}"
                    )

                with col2:
                    st.metric(
                        "📍 Centroid",
                        f"{f_q['centroid_mean']:.0f} Hz"
                    )

                with col3:
                    st.metric(
                        "📊 Bandwidth",
                        f"{f_q['bandwidth_mean']:.0f} Hz"
                    )

            # -----------------------------------------------------------------
            # Show Vector
            # -----------------------------------------------------------------

            if show_vector:

                st.subheader("🔢 Combined 33D Vector")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**7 Scalars (Pitch, ZCR, VR, Energy, Harm, Cent, Band):**")
                    for i, val in enumerate(combined_vector[:7]):
                        st.caption(f"{i}: {val:.4f}")

                with col2:
                    st.write("**13 MFCC Static:**")

                    for i, val in enumerate(combined_vector[7:20]):
                        st.caption(f"{i}: {val:.4f}")

                with col3:
                    st.write("**13 MFCC Std:**")

                    for i, val in enumerate(combined_vector[20:33]):
                        st.caption(f"{i}: {val:.4f}")

            # -----------------------------------------------------------------
            # Search Database
            # -----------------------------------------------------------------

            with st.spinner(f"🔍 Searching top-{top_k} results..."):

                try:

                    with psycopg.connect(get_connection_string()) as conn:

                        with conn.cursor() as cur:

                            cur.execute("""
                                SELECT
                                    f.file_name,
                                    af.pitch_mean,
                                    af.average_energy,
                                    af.zcr_mean,
                                    af.voiced_ratio,
                                    af.harmonicity,
                                    af.centroid_mean,
                                    af.bandwidth_mean,
                                    (
                                        1.0 -
                                        (
                                            af.combined_features_vector
                                            <=>
                                            %s::vector
                                        )
                                    ) AS similarity_score,
                                    f.file_path
                                FROM audio_features af
                                JOIN audio_files f
                                    ON f.id = af.audio_id
                                ORDER BY similarity_score DESC
                                LIMIT %s
                            """, (combined_vector, top_k))

                            results = cur.fetchall()

                except Exception as e:

                    st.error(f"❌ Database Error: {e}")
                    st.stop()

            # -----------------------------------------------------------------
            # No Results
            # -----------------------------------------------------------------

            if not results:

                st.error(
                    "❌ No results found. "
                    "Run extract_features.py first."
                )

                st.stop()

            # -----------------------------------------------------------------
            # Display Results
            # -----------------------------------------------------------------

            st.subheader(f"🏆 Top-{top_k} Results")

            for i, (
                name,
                pitch,
                energy,
                zcr,
                vr,
                harmonicity,
                centroid,
                bandwidth,
                score,
                path
            ) in enumerate(results, start=1):

                with st.expander(
                    f"#{i} - {name} | Similarity: {score:.1%}",
                    expanded=(i == 1)
                ):

                    col1, col2, col3 = st.columns([2, 2, 2])

                    # ---------------------------------------------------------
                    # Audio
                    # ---------------------------------------------------------

                    with col1:

                        st.write(f"**📄 File:** {name}")

                        if os.path.exists(path):
                            st.audio(path)
                        else:
                            st.warning(
                                f"⚠️ File not accessible:\n{path}"
                            )

                    # ---------------------------------------------------------
                    # Metrics
                    # ---------------------------------------------------------

                    with col2:

                        st.metric(
                            "🎵 Pitch",
                            f"{pitch:.0f} Hz"
                        )

                        st.metric(
                            "📊 Energy",
                            f"{energy:.4f}"
                        )

                        st.metric(
                            "🗣️ Voiced%",
                            f"{vr:.1%}"
                        )

                    with col3:

                        st.metric(
                            "📈 ZCR",
                            f"{zcr:.4f}"
                        )

                        st.metric(
                            "🎼 Harmonicity",
                            f"{harmonicity:.4f}"
                        )

                        st.metric(
                            "📍 Centroid",
                            f"{centroid:.0f} Hz"
                        )

                    # ---------------------------------------------------------
                    # Similarity Bar
                    # ---------------------------------------------------------

                    st.progress(
                        float(score),
                        text=f"Similarity: {score:.1%}"
                    )

        else:

            st.info(
                "👆 Click the Search button to find similar audio"
            )

else:

    st.info("📁 Upload a WAV file to begin")
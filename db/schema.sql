DROP VIEW IF EXISTS vw_audio_feature_vectors;
DROP TABLE IF EXISTS retrieval_results;
DROP TABLE IF EXISTS retrieval_sessions;
DROP TABLE IF EXISTS audio_features;
DROP TABLE IF EXISTS audio_files;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS audio_files (
    id BIGSERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    sample_rate INTEGER NOT NULL,
    duration_sec DOUBLE PRECISION NOT NULL,
    bit_depth INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audio_features (
    audio_id BIGINT PRIMARY KEY REFERENCES audio_files(id) ON DELETE CASCADE,
    
    -- Dac trung don le
    average_energy DOUBLE PRECISION,
    zcr_mean DOUBLE PRECISION,
    pitch_mean DOUBLE PRECISION,
    centroid_mean DOUBLE PRECISION,
    bandwidth_mean DOUBLE PRECISION,
    harmonicity DOUBLE PRECISION,
    
    -- Vector dac trung (Tach biet)
    mfcc_static vector(13),
    mfcc_std vector(13),
    
    -- Ket qua phan cum
    cluster_id INTEGER,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE VIEW vw_audio_feature_vectors AS
SELECT
    af.audio_id, f.file_name,
    af.pitch_mean, af.average_energy, af.zcr_mean, 
    af.centroid_mean, af.bandwidth_mean, af.harmonicity,
    af.mfcc_static, af.mfcc_std, af.cluster_id
FROM audio_features af
JOIN audio_files f ON f.id = af.audio_id;

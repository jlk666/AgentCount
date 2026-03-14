CREATE TABLE IF NOT EXISTS plate_analysis_results (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sample_id TEXT NOT NULL,
    replicate_id TEXT,
    image_path TEXT NOT NULL,
    dilution DOUBLE PRECISION,
    plated_volume_ml DOUBLE PRECISION,
    colony_count INTEGER,
    cfu_per_ml DOUBLE PRECISION,
    qc_passed BOOLEAN,
    validation_passed BOOLEAN,
    warnings JSONB,
    result_payload JSONB NOT NULL,
    annotated_image_path TEXT
);

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
    asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id TEXT NOT NULL,
    external_id TEXT NOT NULL,
    root_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    content_sha256 TEXT,
    size INTEGER,
    mtime_ns INTEGER,
    source_revision TEXT,
    asset_version INTEGER NOT NULL DEFAULT 1,
    character_id TEXT,
    album_id TEXT,
    work_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    active_extraction_id TEXT,
    deleted_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(collection_id, external_id)
);

CREATE TABLE IF NOT EXISTS extractions (
    extraction_id TEXT PRIMARY KEY,
    content_sha256 TEXT NOT NULL,
    detector_hash TEXT,
    pose_hash TEXT,
    preprocess_version TEXT NOT NULL,
    keypoint_schema TEXT NOT NULL,
    image_width INTEGER NOT NULL,
    image_height INTEGER NOT NULL,
    status TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS persons (
    person_id TEXT PRIMARY KEY,
    extraction_id TEXT NOT NULL REFERENCES extractions(extraction_id) ON DELETE CASCADE,
    person_index INTEGER NOT NULL,
    bbox_blob BLOB NOT NULL,
    keypoints_blob BLOB NOT NULL,
    raw_scores_blob BLOB NOT NULL,
    valid_mask_blob BLOB NOT NULL,
    geometry_blob BLOB NOT NULL,
    quality TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    available_scopes TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_persons_extraction ON persons(extraction_id);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    next_run_at TEXT,
    lease_expires_at TEXT,
    heartbeat_at TEXT,
    source_revision TEXT,
    asset_version INTEGER,
    result_json TEXT,
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(status, priority, created_at);

CREATE TABLE IF NOT EXISTS temporary_analyses (
    analysis_id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL,
    extraction_id TEXT,
    upload_path TEXT,
    status TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS changes (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('db_schema_version', '1');

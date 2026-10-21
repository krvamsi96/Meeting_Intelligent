-- ==============================================================================
-- PRODUCTION MIGRATION: Create Missing Mixed-Case Tables
-- Database: meity_db_05122024 (PostgreSQL on 10.192.110.24:5432)
-- Run with: psql -U dbadmin -d meity_db_05122024 -f migrate_missing_tables.sql
--
-- ROOT CAUSE:
--   The tables "AI_Meeting_Inteligent_final" and "AI_Meeting_Activty_Logs"
--   use mixed-case (quoted) identifiers. During a prior startup attempt,
--   the TCP connection to the server timed out mid-transaction, rolling back
--   the CREATE TABLE operations. This script re-creates them safely using
--   IF NOT EXISTS — it will NOT affect any existing data.
--
-- SAFETY: This script will NOT drop, truncate, or alter any existing table.
--         It only creates tables and indexes that are MISSING.
-- ==============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

BEGIN;

-- TABLE: AI_Meeting_Activty_Logs ('Activty' typo is INTENTIONAL)
CREATE TABLE IF NOT EXISTS "AI_Meeting_Activty_Logs" (
    id SERIAL NOT NULL,
    meeting_pk INTEGER NOT NULL,
    bharatvc_meeting_id VARCHAR,
    meeting_id VARCHAR,
    processing_id VARCHAR NOT NULL,
    step_name VARCHAR NOT NULL,
    step_order INTEGER NOT NULL DEFAULT 0,
    status VARCHAR NOT NULL DEFAULT 'PENDING',
    message TEXT,
    started_at TIMESTAMP WITHOUT TIME ZONE,
    completed_at TIMESTAMP WITHOUT TIME ZONE,
    duration_seconds FLOAT,
    progress_percent FLOAT DEFAULT 0.0,
    details_json TEXT,
    error_code VARCHAR,
    error_message TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE,
    updated_at TIMESTAMP WITHOUT TIME ZONE,
    PRIMARY KEY (id),
    CONSTRAINT fk_activity_logs_meeting_pk
        FOREIGN KEY(meeting_pk) REFERENCES meetings (id) ON DELETE CASCADE
);

COMMIT;

CREATE INDEX IF NOT EXISTS "ix_AI_Meeting_Activty_Logs_id" ON "AI_Meeting_Activty_Logs" (id);
CREATE INDEX IF NOT EXISTS ix_activity_logs_created ON "AI_Meeting_Activty_Logs" (created_at);
CREATE INDEX IF NOT EXISTS "ix_AI_Meeting_Activty_Logs_meeting_id" ON "AI_Meeting_Activty_Logs" (meeting_id);
CREATE INDEX IF NOT EXISTS ix_activity_logs_bvc_id_status ON "AI_Meeting_Activty_Logs" (bharatvc_meeting_id, status);
CREATE INDEX IF NOT EXISTS "ix_AI_Meeting_Activty_Logs_processing_id" ON "AI_Meeting_Activty_Logs" (processing_id);
CREATE INDEX IF NOT EXISTS ix_activity_logs_meeting_proc ON "AI_Meeting_Activty_Logs" (meeting_pk, processing_id);
CREATE INDEX IF NOT EXISTS "ix_AI_Meeting_Activty_Logs_meeting_pk" ON "AI_Meeting_Activty_Logs" (meeting_pk);
CREATE INDEX IF NOT EXISTS "ix_AI_Meeting_Activty_Logs_bharatvc_meeting_id" ON "AI_Meeting_Activty_Logs" (bharatvc_meeting_id);

BEGIN;

-- TABLE: AI_Meeting_Inteligent_final ('Inteligent' typo is INTENTIONAL)
CREATE TABLE IF NOT EXISTS "AI_Meeting_Inteligent_final" (
    id SERIAL NOT NULL,
    meeting_pk INTEGER NOT NULL,
    meeting_id VARCHAR,
    bharatvc_meeting_id VARCHAR,
    display_id VARCHAR,
    title VARCHAR,
    meeting_date VARCHAR,
    meeting_time VARCHAR,
    duration_seconds FLOAT,
    duration_formatted VARCHAR,
    status VARCHAR,
    participant_count INTEGER,
    participants TEXT,
    speaker_count INTEGER,
    processing_status VARCHAR NOT NULL DEFAULT 'PENDING',
    aggregation_version VARCHAR DEFAULT '1.0',
    transcript TEXT,
    speakers TEXT,
    speaker_duration TEXT,
    agenda TEXT,
    decisions TEXT,
    agenda_decision_mapping TEXT,
    action_items TEXT,
    summary TEXT,
    mom TEXT,
    sentiment TEXT,
    metrics TEXT,
    activity_logs TEXT,
    error_info TEXT,
    json_data JSONB,
    created_at TIMESTAMP WITHOUT TIME ZONE,
    updated_at TIMESTAMP WITHOUT TIME ZONE,
    PRIMARY KEY (id),
    UNIQUE (meeting_pk),
    CONSTRAINT fk_final_meeting_pk
        FOREIGN KEY(meeting_pk) REFERENCES meetings (id) ON DELETE CASCADE
);

COMMIT;

CREATE INDEX IF NOT EXISTS "ix_AI_Meeting_Inteligent_final_id" ON "AI_Meeting_Inteligent_final" (id);
CREATE UNIQUE INDEX IF NOT EXISTS "ix_AI_Meeting_Inteligent_final_meeting_pk" ON "AI_Meeting_Inteligent_final" (meeting_pk);
CREATE INDEX IF NOT EXISTS ix_final_bvc_id ON "AI_Meeting_Inteligent_final" (bharatvc_meeting_id);
CREATE INDEX IF NOT EXISTS ix_final_meeting_id_str ON "AI_Meeting_Inteligent_final" (meeting_id);
CREATE INDEX IF NOT EXISTS ix_final_meeting_status ON "AI_Meeting_Inteligent_final" (meeting_pk, processing_status);
CREATE INDEX IF NOT EXISTS ix_final_display_id ON "AI_Meeting_Inteligent_final" (display_id);

-- VERIFICATION
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN ('AI_Meeting_Inteligent_final', 'AI_Meeting_Activty_Logs')
ORDER BY table_name, ordinal_position;

SELECT 'AI_Meeting_Inteligent_final' AS table_name, COUNT(*) AS row_count FROM "AI_Meeting_Inteligent_final"
UNION ALL
SELECT 'AI_Meeting_Activty_Logs', COUNT(*) FROM "AI_Meeting_Activty_Logs";

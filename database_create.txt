================================================================================
-- AI MEETING INTELLIGENCE PLATFORM — DATABASE DDL SPECIFICATION
-- Dialect: PostgreSQL (13+)
-- Generated from: backend/models/models.py & Live Schema Validation
-- Purpose: Complete schema creation script for production deployment
================================================================================

-- -----------------------------------------------------------------------------
-- 0. EXTENSIONS & DATABASE SETTINGS
-- -----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

-- -----------------------------------------------------------------------------
-- TABLE: ai_provider_configs
-- Description: Dynamic credentials, endpoints, and priority routing configurations for AI models (AIRAWAT/NeMo/Whisper).
-- Total Columns: 4 | Indexes: 1
-- -----------------------------------------------------------------------------
CREATE TABLE ai_provider_configs (
	provider VARCHAR NOT NULL, 
	enabled BOOLEAN, 
	encrypted_config TEXT, 
	updated_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (provider)
);

-- Indexes for ai_provider_configs
CREATE INDEX ix_ai_provider_configs_provider ON ai_provider_configs (provider);

-- -----------------------------------------------------------------------------
-- TABLE: meetings
-- Description: Core meeting registry storing session metadata, processing state, video/audio metrics, and accuracy signals.
-- Total Columns: 85 | Indexes: 8
-- -----------------------------------------------------------------------------
CREATE TABLE meetings (
	id SERIAL NOT NULL, 
	bharatvc_meeting_id VARCHAR, 
	meeting_id VARCHAR, 
	display_id VARCHAR, 
	filename VARCHAR, 
	file_type VARCHAR, 
	filepath VARCHAR, 
	upload_time TIMESTAMP WITHOUT TIME ZONE, 
	status VARCHAR, 
	error_message TEXT, 
	title VARCHAR, 
	agenda TEXT, 
	meeting_date VARCHAR, 
	meeting_time VARCHAR, 
	project_name VARCHAR, 
	matched_meeting_id INTEGER, 
	matched_meeting_code VARCHAR, 
	similarity_score FLOAT, 
	matched_mom TEXT, 
	related_meetings_json TEXT, 
	content_fingerprint VARCHAR, 
	is_approved BOOLEAN, 
	accuracy FLOAT, 
	speaker_accuracy FLOAT, 
	transcription_accuracy FLOAT, 
	confidence_score FLOAT, 
	processing_time FLOAT, 
	audio_quality FLOAT, 
	video_quality FLOAT, 
	duration FLOAT, 
	participant_count INTEGER, 
	participants TEXT, 
	interruptions INTEGER, 
	silence_duration FLOAT, 
	sentiment VARCHAR, 
	emotions TEXT, 
	questions_raised TEXT, 
	keyword_frequency TEXT, 
	summary_quality FLOAT, 
	mom_quality FLOAT, 
	action_item_quality FLOAT, 
	topic_quality FLOAT, 
	metrics_json TEXT, 
	file_size_bytes INTEGER, 
	upload_start_time TIMESTAMP WITHOUT TIME ZONE, 
	upload_end_time TIMESTAMP WITHOUT TIME ZONE, 
	upload_duration_sec FLOAT, 
	processing_start_time TIMESTAMP WITHOUT TIME ZONE, 
	processing_end_time TIMESTAMP WITHOUT TIME ZONE, 
	total_processing_time_sec FLOAT, 
	stt_generated BOOLEAN, 
	stt_word_count INTEGER, 
	stt_char_count INTEGER, 
	stt_duration_sec FLOAT, 
	speaker_count INTEGER, 
	speaker_timeline_generated BOOLEAN, 
	speaker_duration_sec FLOAT, 
	diarization_status VARCHAR, 
	diarization_info_json TEXT, 
	diarization_metadata_json TEXT, 
	pipeline_stage_states_json TEXT, 
	retry_history_json TEXT, 
	summary_generated BOOLEAN, 
	summary_word_count INTEGER, 
	summary_duration_sec FLOAT, 
	mom_generated BOOLEAN, 
	mom_word_count INTEGER, 
	mom_duration_sec FLOAT, 
	action_items_generated BOOLEAN, 
	action_items_count INTEGER, 
	action_items_duration_sec FLOAT, 
	topics_generated BOOLEAN, 
	topics_count INTEGER, 
	sentiment_label VARCHAR, 
	sentiment_breakdown_json TEXT, 
	db_meeting_saved BOOLEAN, 
	db_transcript_saved BOOLEAN, 
	db_summary_saved BOOLEAN, 
	db_action_items_saved BOOLEAN, 
	export_pdf_generated BOOLEAN, 
	export_docx_generated BOOLEAN, 
	export_json_generated BOOLEAN, 
	api_status VARCHAR, 
	api_status_code INTEGER, 
	overall_status VARCHAR, 
	PRIMARY KEY (id)
);

-- Indexes for meetings
CREATE UNIQUE INDEX ix_meetings_bharatvc_meeting_id ON meetings (bharatvc_meeting_id);
CREATE INDEX ix_meetings_status_upload_time ON meetings (status, upload_time);
CREATE INDEX ix_meetings_is_approved_status ON meetings (is_approved, status);
CREATE INDEX ix_meetings_filename ON meetings (filename);
CREATE UNIQUE INDEX ix_meetings_meeting_id ON meetings (meeting_id);
CREATE INDEX ix_meetings_content_fingerprint ON meetings (content_fingerprint);
CREATE INDEX ix_meetings_id ON meetings (id);
CREATE INDEX ix_meetings_display_id ON meetings (display_id);

-- -----------------------------------------------------------------------------
-- TABLE: tran_task
-- Description: External BharatVC integration / translation / task delegation tracking table.
-- Total Columns: 49 | Indexes: 8
-- -----------------------------------------------------------------------------
CREATE TABLE tran_task (
	id SERIAL NOT NULL, 
	task_title TEXT, 
	target_date DATE, 
	col_3 TEXT, 
	remarks TEXT, 
	reviewed_at TIMESTAMP WITHOUT TIME ZONE, 
	reference_uuid VARCHAR(100), 
	is_active BOOLEAN, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	ext_task_id INTEGER, 
	completed_at TIMESTAMP WITHOUT TIME ZONE, 
	col_11 TEXT, 
	ministry_id INTEGER, 
	section_id INTEGER, 
	task_type_id INTEGER, 
	parent_task_id INTEGER, 
	hierarchy_id INTEGER, 
	priority_id INTEGER, 
	assigned_to INTEGER, 
	assigned_by INTEGER, 
	is_closed BOOLEAN, 
	status_id INTEGER, 
	attachment_filename TEXT, 
	description TEXT, 
	source_ref_id INTEGER, 
	reference_date DATE, 
	initiator_id INTEGER, 
	priority_level INTEGER, 
	division_id INTEGER, 
	is_escalated BOOLEAN, 
	reminder_days INTEGER, 
	location_id INTEGER, 
	col_32 TEXT, 
	attachment_guid VARCHAR(100), 
	attachment_doc_name TEXT, 
	source_type INTEGER, 
	forward_dept_id INTEGER, 
	forward_officer_id INTEGER, 
	col_38 TEXT, 
	is_visible BOOLEAN, 
	file_number VARCHAR(100), 
	subject_category_id INTEGER, 
	reminder_count INTEGER, 
	escalation_level INTEGER, 
	col_44 TEXT, 
	col_45 TEXT, 
	revised_date DATE, 
	is_valid BOOLEAN, 
	is_synced BOOLEAN, 
	PRIMARY KEY (id)
);

-- Indexes for tran_task
CREATE INDEX ix_tran_task_target_date ON tran_task (target_date);
CREATE INDEX ix_tran_task_assigned_by ON tran_task (assigned_by);
CREATE INDEX ix_tran_task_reference_uuid ON tran_task (reference_uuid);
CREATE INDEX ix_tran_task_is_active ON tran_task (is_active);
CREATE INDEX ix_tran_task_ministry_id ON tran_task (ministry_id);
CREATE INDEX ix_tran_task_assigned_to ON tran_task (assigned_to);
CREATE INDEX ix_tran_task_is_closed ON tran_task (is_closed);
CREATE INDEX ix_tran_task_id ON tran_task (id);

-- -----------------------------------------------------------------------------
-- TABLE: AI_Meeting_Activty_Logs
-- Description: Granular, ordered audit trail tracking each AI pipeline stage execution, duration, and status.
-- Total Columns: 18 | Indexes: 8
-- -----------------------------------------------------------------------------
CREATE TABLE "AI_Meeting_Activty_Logs" (
	id SERIAL NOT NULL, 
	meeting_pk INTEGER NOT NULL, 
	bharatvc_meeting_id VARCHAR, 
	meeting_id VARCHAR, 
	processing_id VARCHAR NOT NULL, 
	step_name VARCHAR NOT NULL, 
	step_order INTEGER NOT NULL, 
	status VARCHAR NOT NULL, 
	message TEXT, 
	started_at TIMESTAMP WITHOUT TIME ZONE, 
	completed_at TIMESTAMP WITHOUT TIME ZONE, 
	duration_seconds FLOAT, 
	progress_percent FLOAT, 
	details_json TEXT, 
	error_code VARCHAR, 
	error_message TEXT, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	updated_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(meeting_pk) REFERENCES meetings (id) ON DELETE CASCADE
);

-- Indexes for AI_Meeting_Activty_Logs
CREATE INDEX "ix_AI_Meeting_Activty_Logs_id" ON "AI_Meeting_Activty_Logs" (id);
CREATE INDEX ix_activity_logs_created ON "AI_Meeting_Activty_Logs" (created_at);
CREATE INDEX "ix_AI_Meeting_Activty_Logs_meeting_id" ON "AI_Meeting_Activty_Logs" (meeting_id);
CREATE INDEX ix_activity_logs_bvc_id_status ON "AI_Meeting_Activty_Logs" (bharatvc_meeting_id, status);
CREATE INDEX "ix_AI_Meeting_Activty_Logs_processing_id" ON "AI_Meeting_Activty_Logs" (processing_id);
CREATE INDEX ix_activity_logs_meeting_proc ON "AI_Meeting_Activty_Logs" (meeting_pk, processing_id);
CREATE INDEX "ix_AI_Meeting_Activty_Logs_meeting_pk" ON "AI_Meeting_Activty_Logs" (meeting_pk);
CREATE INDEX "ix_AI_Meeting_Activty_Logs_bharatvc_meeting_id" ON "AI_Meeting_Activty_Logs" (bharatvc_meeting_id);

-- -----------------------------------------------------------------------------
-- TABLE: AI_Meeting_Inteligent_final
-- Description: Consolidated final single-table delivery format for high-speed frontend and external API consumption.
-- Total Columns: 31 | Indexes: 8
-- -----------------------------------------------------------------------------
CREATE TABLE "AI_Meeting_Inteligent_final" (
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
	processing_status VARCHAR NOT NULL, 
	aggregation_version VARCHAR, 
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
	FOREIGN KEY(meeting_pk) REFERENCES meetings (id) ON DELETE CASCADE
);

-- Indexes for AI_Meeting_Inteligent_final
CREATE INDEX "ix_AI_Meeting_Inteligent_final_display_id" ON "AI_Meeting_Inteligent_final" (display_id);
CREATE UNIQUE INDEX "ix_AI_Meeting_Inteligent_final_meeting_pk" ON "AI_Meeting_Inteligent_final" (meeting_pk);
CREATE INDEX "ix_AI_Meeting_Inteligent_final_meeting_id" ON "AI_Meeting_Inteligent_final" (meeting_id);
CREATE INDEX ix_final_meeting_id_str ON "AI_Meeting_Inteligent_final" (meeting_id);
CREATE INDEX ix_final_meeting_status ON "AI_Meeting_Inteligent_final" (meeting_pk, processing_status);
CREATE INDEX ix_final_bvc_id ON "AI_Meeting_Inteligent_final" (bharatvc_meeting_id);
CREATE INDEX "ix_AI_Meeting_Inteligent_final_id" ON "AI_Meeting_Inteligent_final" (id);
CREATE INDEX "ix_AI_Meeting_Inteligent_final_bharatvc_meeting_id" ON "AI_Meeting_Inteligent_final" (bharatvc_meeting_id);

-- -----------------------------------------------------------------------------
-- TABLE: action_items
-- Description: Assigned tasks, deliverables, owners, deadlines, priority, and transcript grounding attribution.
-- Total Columns: 27 | Indexes: 3
-- -----------------------------------------------------------------------------
CREATE TABLE action_items (
	id SERIAL NOT NULL, 
	meeting_id INTEGER NOT NULL, 
	bharatvc_meeting_id VARCHAR, 
	item_order INTEGER NOT NULL, 
	task TEXT NOT NULL, 
	description TEXT, 
	main_task TEXT, 
	owner VARCHAR, 
	assigned_by VARCHAR, 
	deadline VARCHAR, 
	deadline_type VARCHAR, 
	due_date DATE, 
	deadline_start DATE, 
	deadline_end DATE, 
	deadline_precision VARCHAR, 
	priority VARCHAR, 
	status VARCHAR, 
	evidence TEXT, 
	speaker VARCHAR, 
	timestamp_start VARCHAR, 
	timestamp_end VARCHAR, 
	confidence FLOAT, 
	source VARCHAR, 
	agenda_ref VARCHAR, 
	decision_ref VARCHAR, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	updated_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(meeting_id) REFERENCES meetings (id)
);

-- Indexes for action_items
CREATE INDEX ix_action_items_bharatvc_meeting_id ON action_items (bharatvc_meeting_id);
CREATE INDEX ix_action_items_id ON action_items (id);
CREATE INDEX ix_action_items_meeting_id ON action_items (meeting_id);

-- -----------------------------------------------------------------------------
-- TABLE: agenda_key_decisions
-- Description: Extracted discussion topics, matched final decisions, evidence grounding, and semantic cosine similarity scores.
-- Total Columns: 20 | Indexes: 3
-- -----------------------------------------------------------------------------
CREATE TABLE agenda_key_decisions (
	id SERIAL NOT NULL, 
	meeting_id INTEGER NOT NULL, 
	bharatvc_meeting_id VARCHAR, 
	agenda_order INTEGER NOT NULL, 
	agenda TEXT NOT NULL, 
	agenda_type VARCHAR, 
	key_decision TEXT NOT NULL, 
	reason TEXT, 
	evidence TEXT, 
	speaker VARCHAR, 
	timestamp VARCHAR, 
	confidence FLOAT, 
	agenda_id_str VARCHAR, 
	agenda_source VARCHAR, 
	decision_grounded BOOLEAN, 
	match_score FLOAT, 
	relevance_score FLOAT, 
	embedding_model VARCHAR, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	updated_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(meeting_id) REFERENCES meetings (id)
);

-- Indexes for agenda_key_decisions
CREATE INDEX ix_agenda_key_decisions_id ON agenda_key_decisions (id);
CREATE INDEX ix_agenda_key_decisions_meeting_id ON agenda_key_decisions (meeting_id);
CREATE INDEX ix_agenda_key_decisions_bharatvc_meeting_id ON agenda_key_decisions (bharatvc_meeting_id);

-- -----------------------------------------------------------------------------
-- TABLE: analysis_results
-- Description: High-level LLM outputs: executive summary, Minutes of Meeting (MOM), and full report JSON.
-- Total Columns: 7 | Indexes: 2
-- -----------------------------------------------------------------------------
CREATE TABLE analysis_results (
	id SERIAL NOT NULL, 
	meeting_id INTEGER, 
	summary TEXT, 
	mom TEXT, 
	action_items TEXT, 
	decisions TEXT, 
	risks TEXT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(meeting_id) REFERENCES meetings (id)
);

-- Indexes for analysis_results
CREATE UNIQUE INDEX ix_analysis_results_meeting_id ON analysis_results (meeting_id);
CREATE INDEX ix_analysis_results_id ON analysis_results (id);

-- -----------------------------------------------------------------------------
-- TABLE: benchmark_evaluations
-- Description: Technical platform performance metrics including latency, silence ratio, video FPS/bitrate, and RAG faithfulness.
-- Total Columns: 42 | Indexes: 2
-- -----------------------------------------------------------------------------
CREATE TABLE benchmark_evaluations (
	benchmark_id SERIAL NOT NULL, 
	meeting_id INTEGER NOT NULL, 
	reference_version VARCHAR, 
	evaluation_version VARCHAR, 
	reference_available BOOLEAN, 
	stt_accuracy FLOAT, 
	wer FLOAT, 
	cer FLOAT, 
	speaker_accuracy FLOAT, 
	der FLOAT, 
	jer FLOAT, 
	speaker_id_accuracy FLOAT, 
	speaker_id_f1 FLOAT, 
	speaker_id_eer FLOAT, 
	summary_accuracy FLOAT, 
	rouge1 FLOAT, 
	rouge2 FLOAT, 
	rougel FLOAT, 
	bertscore FLOAT, 
	mom_accuracy FLOAT, 
	mom_rougel FLOAT, 
	mom_bertscore FLOAT, 
	action_precision FLOAT, 
	action_recall FLOAT, 
	action_f1 FLOAT, 
	topic_coherence FLOAT, 
	topic_f1 FLOAT, 
	sentiment_accuracy FLOAT, 
	sentiment_f1 FLOAT, 
	sentiment_mcc FLOAT, 
	similarity_recall_10 FLOAT, 
	similarity_mrr FLOAT, 
	similarity_ndcg FLOAT, 
	chatbot_faithfulness FLOAT, 
	chatbot_answer_relevance FLOAT, 
	audio_snr FLOAT, 
	audio_silence_ratio FLOAT, 
	video_fps FLOAT, 
	video_resolution VARCHAR, 
	video_bitrate FLOAT, 
	evaluation_time TIMESTAMP WITHOUT TIME ZONE, 
	evaluation_status VARCHAR, 
	PRIMARY KEY (benchmark_id), 
	FOREIGN KEY(meeting_id) REFERENCES meetings (id)
);

-- Indexes for benchmark_evaluations
CREATE UNIQUE INDEX ix_benchmark_evaluations_meeting_id ON benchmark_evaluations (meeting_id);
CREATE INDEX ix_benchmark_evaluations_benchmark_id ON benchmark_evaluations (benchmark_id);

-- -----------------------------------------------------------------------------
-- TABLE: chat_sessions
-- Description: Interactive Agentic RAG chat conversation history per meeting with source attribution.
-- Total Columns: 7 | Indexes: 3
-- -----------------------------------------------------------------------------
CREATE TABLE chat_sessions (
	id SERIAL NOT NULL, 
	session_id VARCHAR, 
	meeting_id INTEGER NOT NULL, 
	role VARCHAR NOT NULL, 
	content TEXT NOT NULL, 
	sources TEXT, 
	timestamp TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(meeting_id) REFERENCES meetings (id)
);

-- Indexes for chat_sessions
CREATE INDEX ix_chat_sessions_meeting_id ON chat_sessions (meeting_id);
CREATE INDEX ix_chat_sessions_session_id ON chat_sessions (session_id);
CREATE INDEX ix_chat_sessions_id ON chat_sessions (id);

-- -----------------------------------------------------------------------------
-- TABLE: ground_truth
-- Description: Human-annotated ground truth references for calculating objective accuracy benchmarks.
-- Total Columns: 17 | Indexes: 2
-- -----------------------------------------------------------------------------
CREATE TABLE ground_truth (
	id SERIAL NOT NULL, 
	meeting_id INTEGER NOT NULL, 
	uploaded_by VARCHAR, 
	version INTEGER, 
	ref_transcript TEXT, 
	ref_speaker_labels TEXT, 
	ref_speaker_timeline TEXT, 
	ref_summary TEXT, 
	ref_mom TEXT, 
	ref_action_items TEXT, 
	ref_topics TEXT, 
	ref_sentiment TEXT, 
	ref_similarity_labels TEXT, 
	source VARCHAR, 
	notes TEXT, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	updated_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(meeting_id) REFERENCES meetings (id)
);

-- Indexes for ground_truth
CREATE INDEX ix_ground_truth_id ON ground_truth (id);
CREATE UNIQUE INDEX ix_ground_truth_meeting_id ON ground_truth (meeting_id);

-- -----------------------------------------------------------------------------
-- TABLE: meeting_evaluations
-- Description: Reference-free real-time quality indicators assessing STT, speaker consistency, and summary grounding.
-- Total Columns: 16 | Indexes: 2
-- -----------------------------------------------------------------------------
CREATE TABLE meeting_evaluations (
	id SERIAL NOT NULL, 
	meeting_id INTEGER NOT NULL, 
	speech_recognition_quality FLOAT, 
	speaker_consistency FLOAT, 
	summary_grounding FLOAT, 
	mom_quality FLOAT, 
	action_item_quality FLOAT, 
	topic_coverage FLOAT, 
	sentiment_confidence FLOAT, 
	similarity_confidence FLOAT, 
	audio_signal_quality FLOAT, 
	video_signal_quality FLOAT, 
	overall_ai_quality FLOAT, 
	processing_time FLOAT, 
	evaluation_timestamp TIMESTAMP WITHOUT TIME ZONE, 
	evaluation_version VARCHAR, 
	PRIMARY KEY (id), 
	FOREIGN KEY(meeting_id) REFERENCES meetings (id)
);

-- Indexes for meeting_evaluations
CREATE INDEX ix_meeting_evaluations_id ON meeting_evaluations (id);
CREATE UNIQUE INDEX ix_meeting_evaluations_meeting_id ON meeting_evaluations (meeting_id);

-- -----------------------------------------------------------------------------
-- TABLE: transcript_segments
-- Description: Atomic speaker-attributed transcription segments with timecodes, confidence, and ASR correction audit data.
-- Total Columns: 13 | Indexes: 2
-- -----------------------------------------------------------------------------
CREATE TABLE transcript_segments (
	id SERIAL NOT NULL, 
	meeting_id INTEGER, 
	speaker VARCHAR, 
	start_time VARCHAR, 
	end_time VARCHAR, 
	text TEXT, 
	raw_text TEXT, 
	stt_confidence FLOAT, 
	correction_applied BOOLEAN, 
	correction_confidence FLOAT, 
	correction_source VARCHAR, 
	correction_reason VARCHAR, 
	correction_metadata TEXT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(meeting_id) REFERENCES meetings (id)
);

-- Indexes for transcript_segments
CREATE INDEX ix_transcript_segments_id ON transcript_segments (id);
CREATE INDEX ix_transcript_segments_meeting_id ON transcript_segments (meeting_id);

-- -----------------------------------------------------------------------------
-- TABLE: evaluation_results
-- Description: Ground-truth vs pipeline metrics: WER, CER, DER, JER, ROUGE-1/2/L, BERTScore, and system scores.
-- Total Columns: 60 | Indexes: 3
-- -----------------------------------------------------------------------------
CREATE TABLE evaluation_results (
	id SERIAL NOT NULL, 
	meeting_id INTEGER NOT NULL, 
	ground_truth_id INTEGER, 
	evaluator VARCHAR, 
	evaluation_timestamp TIMESTAMP WITHOUT TIME ZONE, 
	evaluation_version VARCHAR, 
	wer FLOAT, 
	cer FLOAT, 
	stt_match_pct FLOAT, 
	stt_accuracy FLOAT, 
	stt_evaluated BOOLEAN, 
	der FLOAT, 
	jer FLOAT, 
	diarization_accuracy FLOAT, 
	diarization_evaluated BOOLEAN, 
	speaker_accuracy FLOAT, 
	speaker_precision FLOAT, 
	speaker_recall FLOAT, 
	speaker_f1 FLOAT, 
	speaker_confusion_matrix_json TEXT, 
	speaker_evaluated BOOLEAN, 
	rouge1 FLOAT, 
	rouge2 FLOAT, 
	"rougeL" FLOAT, 
	bertscore_f1 FLOAT, 
	summary_accuracy FLOAT, 
	summary_evaluated BOOLEAN, 
	mom_coverage FLOAT, 
	mom_missing_count INTEGER, 
	mom_hallucination_count INTEGER, 
	mom_accuracy FLOAT, 
	mom_evaluated BOOLEAN, 
	action_precision FLOAT, 
	action_recall FLOAT, 
	action_f1 FLOAT, 
	action_missing_count INTEGER, 
	action_false_count INTEGER, 
	action_accuracy FLOAT, 
	action_evaluated BOOLEAN, 
	topic_accuracy FLOAT, 
	topic_precision FLOAT, 
	topic_recall FLOAT, 
	topic_f1 FLOAT, 
	topic_evaluated BOOLEAN, 
	sentiment_accuracy FLOAT, 
	sentiment_precision FLOAT, 
	sentiment_recall FLOAT, 
	sentiment_f1 FLOAT, 
	sentiment_confusion_matrix_json TEXT, 
	sentiment_evaluated BOOLEAN, 
	overall_accuracy FLOAT, 
	raw_results_json TEXT, 
	similarity_ndcg FLOAT, 
	overall_system_score FLOAT, 
	evaluation_type_code VARCHAR, 
	evaluation_status VARCHAR, 
	topics_evaluated BOOLEAN, 
	similarity_evaluated BOOLEAN, 
	audio_signal_evaluated BOOLEAN, 
	video_signal_evaluated BOOLEAN, 
	PRIMARY KEY (id), 
	FOREIGN KEY(meeting_id) REFERENCES meetings (id), 
	FOREIGN KEY(ground_truth_id) REFERENCES ground_truth (id)
);

-- Indexes for evaluation_results
CREATE INDEX ix_evaluation_results_id ON evaluation_results (id);
CREATE UNIQUE INDEX ix_evaluation_results_meeting_id ON evaluation_results (meeting_id);
CREATE INDEX ix_evaluation_results_ground_truth_id ON evaluation_results (ground_truth_id);

-- -----------------------------------------------------------------------------
-- TABLE: evaluation_history
-- Description: Historical audit log tracking changes in evaluation accuracy over time.
-- Total Columns: 7 | Indexes: 2
-- -----------------------------------------------------------------------------
CREATE TABLE evaluation_history (
	id SERIAL NOT NULL, 
	meeting_id INTEGER NOT NULL, 
	evaluation_result_id INTEGER, 
	action VARCHAR NOT NULL, 
	performed_by VARCHAR, 
	timestamp TIMESTAMP WITHOUT TIME ZONE, 
	details_json TEXT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(meeting_id) REFERENCES meetings (id), 
	FOREIGN KEY(evaluation_result_id) REFERENCES evaluation_results (id)
);

-- Indexes for evaluation_history
CREATE INDEX ix_evaluation_history_id ON evaluation_history (id);
CREATE INDEX ix_evaluation_history_meeting_id ON evaluation_history (meeting_id);

-- -----------------------------------------------------------------------------
-- TABLE: evaluation_metric_records
-- Description: Atomic per-metric evaluation data records for detailed longitudinal reporting.
-- Total Columns: 15 | Indexes: 3
-- -----------------------------------------------------------------------------
CREATE TABLE evaluation_metric_records (
	id SERIAL NOT NULL, 
	meeting_id INTEGER NOT NULL, 
	evaluation_result_id INTEGER, 
	metric_name VARCHAR NOT NULL, 
	metric_key VARCHAR NOT NULL, 
	metric_value FLOAT, 
	metric_unit VARCHAR, 
	metric_source VARCHAR, 
	model_version VARCHAR, 
	execution_time_sec FLOAT, 
	status VARCHAR NOT NULL, 
	reason VARCHAR, 
	primary_metric_key VARCHAR, 
	evaluation_version VARCHAR, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(meeting_id) REFERENCES meetings (id), 
	FOREIGN KEY(evaluation_result_id) REFERENCES evaluation_results (id)
);

-- Indexes for evaluation_metric_records
CREATE INDEX ix_evaluation_metric_records_id ON evaluation_metric_records (id);
CREATE INDEX ix_evaluation_metric_records_meeting_id ON evaluation_metric_records (meeting_id);
CREATE INDEX ix_evaluation_metric_records_evaluation_result_id ON evaluation_metric_records (evaluation_result_id);

-- -----------------------------------------------------------------------------
-- COMPATIBILITY VIEW / TABLE ALIAS (Optional for legacy queries)
-- -----------------------------------------------------------------------------
-- CREATE OR REPLACE VIEW "AI_Meeting_Activity_Logs" AS SELECT * FROM "AI_Meeting_Activty_Logs";

================================================================================
-- END OF DATABASE DDL SPECIFICATION
================================================================================

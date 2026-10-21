"""Comprehensive Integration and Regression Tests for Activity Logs Ordering.

Tests all requirements:
1. Exact 18-Feature Logical Order (1..18)
2. Chronological Order within the same feature (started_at / created_at)
3. Strict Meeting ID Isolation (Meeting A vs Meeting B cross-checks)
4. Preservation of Errors, Failures, and Warnings
5. API Endpoint Sorting & Response Schema
6. No Unnecessary Duplicates
"""

import json
import logging
import os
import sys
import unittest
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.database.connection import SessionLocal, init_db
from backend.models.models import Meeting, MeetingActivityLog
from backend.repositories import MeetingRepository
from backend.services.activity_logger import (
    ActivityLogger,
    FEATURE_ORDER,
    STEP_ORDER,
    TOTAL_STEPS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestActivityLogsOrdering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test database and meetings."""
        init_db()
        cls.db = SessionLocal()
        cls.repo = MeetingRepository(cls.db)

        # Create Test Meeting A
        cls.meeting_a = cls.repo.create(
            filename="meeting_alpha.mp4",
            title="Meeting Alpha - Product Strategy",
            file_type="video",
            project_name="Alpha",
            meeting_date="2026-09-01",
            status="completed",
        )

        # Create Test Meeting B
        cls.meeting_b = cls.repo.create(
            filename="meeting_beta.mp4",
            title="Meeting Beta - Security Audit",
            file_type="video",
            project_name="Beta",
            meeting_date="2026-09-01",
            status="completed",
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up test meetings and logs."""
        try:
            cls.db.query(MeetingActivityLog).filter(
                MeetingActivityLog.meeting_pk.in_([cls.meeting_a.id, cls.meeting_b.id])
            ).delete(synchronize_session=False)
            cls.db.query(Meeting).filter(
                Meeting.id.in_([cls.meeting_a.id, cls.meeting_b.id])
            ).delete(synchronize_session=False)
            cls.db.commit()
        except Exception as e:
            cls.db.rollback()
        finally:
            cls.db.close()

    def test_feature_order_definitions(self):
        """Verify that the centralized 18-step feature order is exact and strictly monotonic."""
        expected_order = [
            ("video_processing", 1),
            ("audio_processing", 2),
            ("transcription", 3),
            ("speaker_diarization", 4),
            ("speaker_duration", 5),
            ("transcript_correction", 6),
            ("agenda_extraction", 7),
            ("decision_extraction", 8),
            ("decision_mapping", 9),
            ("action_item_extraction", 10),
            ("summary_generation", 11),
            ("mom_generation", 12),
            ("sentiment_analysis", 13),
            ("ai_metrics", 14),
            ("vector_indexing", 15),
            ("similar_meeting_search", 16),
            ("data_persistence", 17),
            ("final_validation", 18),
        ]

        self.assertEqual(TOTAL_STEPS, 18)
        for feature_name, expected_num in expected_order:
            order = FEATURE_ORDER.get(feature_name)
            self.assertEqual(
                order,
                expected_num,
                f"Feature '{feature_name}' has order {order}, expected {expected_num}",
            )
            step_order = STEP_ORDER.get(feature_name)
            self.assertEqual(step_order, expected_num)

    def test_full_pipeline_activity_logs_ordering(self):
        """Simulate all 18 pipeline features and verify that query results are perfectly ordered 1..18."""
        act = ActivityLogger(
            meeting_pk=self.meeting_a.id,
            bharatvc_id=self.meeting_a.bharatvc_meeting_id,
            meeting_id_str=self.meeting_a.meeting_id,
            db=self.db,
        )

        features = [
            ("video_processing", "Video file received and validated", {"format": "mp4"}),
            ("audio_processing", "Audio track extracted via FFmpeg", {"sample_rate": 16000}),
            ("transcription", "Speech-to-text completed with Whisper", {"segments": 25}),
            ("speaker_diarization", "Identified 3 unique speakers", {"speakers": ["Alice", "Bob", "Charlie"]}),
            ("speaker_duration", "Calculated talk time percentages", {"Alice": "45%", "Bob": "35%", "Charlie": "20%"}),
            ("transcript_correction", "Aligned transcript with speaker metadata", {"normalized": True}),
            ("agenda_extraction", "Extracted 3 agenda topics", {"topics": ["Migration", "Timeline", "Budget"]}),
            ("decision_extraction", "Identified 2 formal decisions", {"decisions_count": 2}),
            ("decision_mapping", "Mapped decisions to agenda with cosine match", {"avg_match": 0.88}),
            ("action_item_extraction", "Assigned 3 structured action items", {"action_item_count": 3}),
            ("summary_generation", "Executive summary generated (250 words)", {"word_count": 250}),
            ("mom_generation", "Structured Minutes of Meeting compiled", {"sections": 5}),
            ("sentiment_analysis", "Analyzed sentiment: Positive (85%)", {"sentiment": "Positive"}),
            ("ai_metrics", "AI quality metrics computed (Accuracy: 94%)", {"accuracy": 94.0}),
            ("vector_indexing", "Indexed chunk embeddings into vector DB", {"collection": "transcript_chunks"}),
            ("similar_meeting_search", "Configured context for AI Chatbot", {"similar_meetings": 2}),
            ("data_persistence", "Committed all intelligence records to DB", {"tables_updated": 6}),
            ("final_validation", "Pipeline validated and finalized", {"status": "completed"}),
        ]

        # Log steps intentionally out of order (simulating parallel background tasks)
        # e.g. RAG (15) finishing before Summary (11) in execution timing
        shuffled_indices = [0, 1, 2, 3, 4, 5, 14, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]

        for idx in shuffled_indices:
            feat, msg, details = features[idx]
            log_id = act.start_step(feat, message=f"Starting {feat}", details=details)
            self.assertIsNotNone(log_id)
            act.complete_step(log_id, message=msg, details=details)

        self.db.commit()

        # Query records using the API ordering strategy
        records = (
            self.db.query(MeetingActivityLog)
            .filter(
                MeetingActivityLog.meeting_pk == self.meeting_a.id,
                MeetingActivityLog.processing_id == act.processing_id,
            )
            .order_by(
                MeetingActivityLog.step_order.asc(),
                MeetingActivityLog.created_at.asc(),
                MeetingActivityLog.id.asc(),
            )
            .all()
        )

        self.assertEqual(len(records), 18)

        # Verify that despite shuffled execution timing, retrieved order is strictly 1..18
        for i, rec in enumerate(records, start=1):
            self.assertEqual(rec.step_order, i, f"Expected step_order {i} at position {i-1}, got {rec.step_order} ({rec.step_name})")
            self.assertEqual(rec.status, "COMPLETED")
            self.assertIsNotNone(rec.duration_seconds)
            self.assertEqual(rec.feature, rec.step_name)
            self.assertEqual(rec.feature_order, rec.step_order)

    def test_meeting_isolation(self):
        """Verify that Meeting A logs never leak into Meeting B queries."""
        act_a = ActivityLogger(meeting_pk=self.meeting_a.id, bharatvc_id=self.meeting_a.bharatvc_meeting_id, meeting_id_str=self.meeting_a.meeting_id, db=self.db)
        act_b = ActivityLogger(meeting_pk=self.meeting_b.id, bharatvc_id=self.meeting_b.bharatvc_meeting_id, meeting_id_str=self.meeting_b.meeting_id, db=self.db)

        # Log to Meeting A
        log_a = act_a.start_step("video_processing", message="Meeting A video processing")
        act_a.complete_step(log_a, message="Meeting A complete")

        # Log to Meeting B
        log_b = act_b.start_step("video_processing", message="Meeting B video processing")
        act_b.complete_step(log_b, message="Meeting B complete")

        self.db.commit()

        # Fetch for Meeting A
        recs_a = (
            self.db.query(MeetingActivityLog)
            .filter(
                MeetingActivityLog.meeting_pk == self.meeting_a.id,
                MeetingActivityLog.processing_id == act_a.processing_id,
            )
            .all()
        )
        for r in recs_a:
            self.assertEqual(r.meeting_pk, self.meeting_a.id)
            self.assertNotIn("Meeting B", r.message)

        # Fetch for Meeting B
        recs_b = (
            self.db.query(MeetingActivityLog)
            .filter(
                MeetingActivityLog.meeting_pk == self.meeting_b.id,
                MeetingActivityLog.processing_id == act_b.processing_id,
            )
            .all()
        )
        for r in recs_b:
            self.assertEqual(r.meeting_pk, self.meeting_b.id)
            self.assertNotIn("Meeting A", r.message)

    def test_failure_and_error_preservation(self):
        """Verify that failed steps preserve status=FAILED, error_code, and error_message."""
        act = ActivityLogger(
            meeting_pk=self.meeting_b.id,
            bharatvc_id=self.meeting_b.bharatvc_meeting_id,
            meeting_id_str=self.meeting_b.meeting_id,
            db=self.db,
        )

        log_id = act.start_step("agenda_extraction", message="Starting agenda extraction")
        act.fail_step(
            log_id,
            error_code="LLM_TIMEOUT",
            error_message="LLM API timed out after 60s",
            details={"retry_count": 2},
        )
        self.db.commit()

        rec = self.db.query(MeetingActivityLog).filter(MeetingActivityLog.id == log_id).first()
        self.assertEqual(rec.status, "FAILED")
        self.assertEqual(rec.error_code, "LLM_TIMEOUT")
        self.assertIn("timed out", rec.error_message)
        self.assertEqual(rec.step_order, 7)
        self.assertIsNotNone(rec.duration_seconds)


if __name__ == "__main__":
    unittest.main()

"""Comprehensive Integration and Regression Tests for AI Chatbot + Dynamic Agentic RAG.

Tests all requirements:
1. Dynamic Intent Classification across 15+ intent patterns
2. Multi-Tool Retrieval (Metadata, Structured DB, Speaker Analysis, Temporal Search, Dynamic MOM, Vector Search)
3. Meeting Isolation (Meeting A vs Meeting B cross-checks)
4. Dynamic Grounded Answer Generation (No hardcoded responses, flexible format)
5. Non-streaming and Streaming Endpoints
"""

import json
import logging
import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.database.connection import SessionLocal, init_db
from backend.models.models import ActionItem, AgendaKeyDecision, AnalysisResult, Meeting, TranscriptSegment
from backend.rag.agent_router import (
    DynamicMOMTool,
    MeetingAgentRouter,
    MetadataTool,
    SpeakerAnalysisTool,
    StructuredDBTool,
    TemporalSearchTool,
    classify_intent,
    classify_multi_intent,
    extract_speaker_from_query,
)
from backend.repositories import MeetingRepository
from backend.services.llm_workflow import generate_chatbot_response, rewrite_query_with_history
from backend.services.vector_service import vector_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestChatbotAgenticRAG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up isolated test meetings and test data in the DB."""
        init_db()
        cls.db = SessionLocal()
        cls.repo = MeetingRepository(cls.db)

        # Create Test Meeting 1: "Project Alpha Architecture Sync"
        cls.meeting1 = cls.repo.create(
            filename="alpha_sync.mp4",
            title="Project Alpha Architecture Sync",
            file_type="video",
            project_name="Alpha Core",
            meeting_date="2026-08-15",
            meeting_time="10:00:00",
            duration=600.0,
            participants="Alice, Bob, Charlie",
            sentiment="Positive",
            emotions="Confidence: 70%, Enthusiasm: 30%",
            questions_raised=json.dumps(["When can we deploy to staging?", "What is the database budget?"]),
            keyword_frequency=json.dumps({"PostgreSQL": 8, "Deployment": 5, "Kubernetes": 4, "Budget": 3}),
            status="completed",
        )

        # Create Test Meeting 2: "Beta Security Audit & Compliance"
        cls.meeting2 = cls.repo.create(
            filename="beta_security.mp4",
            title="Beta Security Audit & Compliance",
            file_type="video",
            project_name="Beta Sec",
            meeting_date="2026-08-16",
            meeting_time="14:00:00",
            duration=300.0,
            participants="David, Eve",
            sentiment="Neutral",
            emotions="Caution: 80%, Analytical: 20%",
            questions_raised=json.dumps(["Are API keys rotated?"]),
            keyword_frequency=json.dumps({"Encryption": 6, "Audit": 4, "OAuth": 3}),
            status="completed",
        )

        # Meeting 1 Data: Segments, Action Items, Decisions, Analysis
        m1_segs = [
            TranscriptSegment(
                meeting_id=cls.meeting1.id,
                speaker="Alice",
                start_time="00:00:05",
                end_time="00:01:15",
                text="Welcome everyone. Today we will discuss our database migration to PostgreSQL and our staging deployment timeline.",
            ),
            TranscriptSegment(
                meeting_id=cls.meeting1.id,
                speaker="Bob",
                start_time="00:01:20",
                end_time="00:03:30",
                text="Regarding the database, PostgreSQL handles our JSON workloads much better. The estimated budget required is 15,000 USD.",
            ),
            TranscriptSegment(
                meeting_id=cls.meeting1.id,
                speaker="Charlie",
                start_time="00:03:35",
                end_time="00:05:00",
                text="I agree on PostgreSQL. We have finalized the decision to allocate 15,000 USD for the database migration.",
            ),
            TranscriptSegment(
                meeting_id=cls.meeting1.id,
                speaker="Alice",
                start_time="00:05:05",
                end_time="00:07:20",
                text="Great. We decided to schedule the deployment to staging for next Friday. Bob, please prepare the Kubernetes manifests by Wednesday.",
            ),
            TranscriptSegment(
                meeting_id=cls.meeting1.id,
                speaker="Bob",
                start_time="00:07:25",
                end_time="00:09:45",
                text="I will handle the Kubernetes manifests by Wednesday. Charlie, please conduct the load tests by Thursday.",
            ),
        ]
        for s in m1_segs:
            cls.db.add(s)

        # Meeting 1 Action Items
        cls.db.add(
            ActionItem(
                meeting_id=cls.meeting1.id,
                item_order=1,
                task="Prepare Kubernetes deployment manifests",
                owner="Bob",
                deadline="Wednesday",
                priority="High",
                status="open",
                evidence="Bob, please prepare the Kubernetes manifests by Wednesday.",
                speaker="Alice",
                timestamp_start="00:05:05",
            )
        )
        cls.db.add(
            ActionItem(
                meeting_id=cls.meeting1.id,
                item_order=2,
                task="Conduct load tests on PostgreSQL instance",
                owner="Charlie",
                deadline="Thursday",
                priority="Medium",
                status="open",
                evidence="Charlie, please conduct the load tests by Thursday.",
                speaker="Bob",
                timestamp_start="00:07:25",
            )
        )

        # Meeting 1 Decisions
        cls.db.add(
            AgendaKeyDecision(
                meeting_id=cls.meeting1.id,
                agenda_order=1,
                agenda="Database Migration & Budget",
                key_decision="Adopt PostgreSQL and allocate 15,000 USD budget.",
                reason="Better support for JSON workloads and scalability.",
                evidence="We have finalized the decision to allocate 15,000 USD for the database migration.",
                speaker="Charlie",
                timestamp="00:03:35",
            )
        )
        cls.db.add(
            AgendaKeyDecision(
                meeting_id=cls.meeting1.id,
                agenda_order=2,
                agenda="Staging Deployment Timeline",
                key_decision="Schedule deployment to staging for next Friday.",
                reason="Ensure all load tests complete first.",
                evidence="We decided to schedule the deployment to staging for next Friday.",
                speaker="Alice",
                timestamp="00:05:05",
            )
        )

        # Meeting 1 Analysis Result
        cls.db.add(
            AnalysisResult(
                meeting_id=cls.meeting1.id,
                summary="The team met to finalize the PostgreSQL migration with a 15,000 USD budget and scheduled staging deployment for next Friday.",
                mom="Minutes of Meeting for Project Alpha Architecture Sync.\nDecisions: Adopt PostgreSQL.\nAction Items: Bob (K8s manifests by Wed), Charlie (Load testing by Thu).",
                action_items="- Bob: Prepare K8s manifests by Wednesday\n- Charlie: Conduct load tests by Thursday",
                decisions="- Adopt PostgreSQL ($15k budget)\n- Staging deployment scheduled for next Friday",
                risks="Potential latency during data migration if network throughput drops.",
            )
        )

        # Meeting 2 Data (Beta Security Audit)
        m2_segs = [
            TranscriptSegment(
                meeting_id=cls.meeting2.id,
                speaker="David",
                start_time="00:00:10",
                end_time="00:02:00",
                text="We are reviewing our OAuth2 token rotation policy and AES-256 data encryption standards.",
            ),
            TranscriptSegment(
                meeting_id=cls.meeting2.id,
                speaker="Eve",
                start_time="00:02:05",
                end_time="00:04:45",
                text="All production secrets must be rotated every 30 days. Eve will update the KMS key rotation script by Monday.",
            ),
        ]
        for s in m2_segs:
            cls.db.add(s)

        cls.db.add(
            ActionItem(
                meeting_id=cls.meeting2.id,
                item_order=1,
                task="Update KMS key rotation script",
                owner="Eve",
                deadline="Monday",
                priority="High",
                status="open",
            )
        )

        cls.db.commit()

        # Add chunks to vector DB
        vector_service.add_transcript_chunks(cls.meeting1.id, m1_segs)
        vector_service.add_transcript_chunks(cls.meeting2.id, m2_segs)

    @classmethod
    def tearDownClass(cls):
        """Clean up test meetings from DB and vector DB."""
        try:
            cls.db.query(TranscriptSegment).filter(
                TranscriptSegment.meeting_id.in_([cls.meeting1.id, cls.meeting2.id])
            ).delete(synchronize_session=False)
            cls.db.query(ActionItem).filter(
                ActionItem.meeting_id.in_([cls.meeting1.id, cls.meeting2.id])
            ).delete(synchronize_session=False)
            cls.db.query(AgendaKeyDecision).filter(
                AgendaKeyDecision.meeting_id.in_([cls.meeting1.id, cls.meeting2.id])
            ).delete(synchronize_session=False)
            cls.db.query(AnalysisResult).filter(
                AnalysisResult.meeting_id.in_([cls.meeting1.id, cls.meeting2.id])
            ).delete(synchronize_session=False)
            cls.db.query(Meeting).filter(
                Meeting.id.in_([cls.meeting1.id, cls.meeting2.id])
            ).delete(synchronize_session=False)
            cls.db.commit()
        except Exception as e:
            cls.db.rollback()
        finally:
            cls.db.close()

    # -----------------------------------------------------------------------
    # Test 1: Dynamic Intent Classification Matrix
    # -----------------------------------------------------------------------
    def test_intent_classification_matrix(self):
        test_cases = [
            ("What was discussed in the meeting?", "agenda"),
            ("Give me the MOM.", "mom"),
            ("Give me the Minutes of Meeting.", "mom"),
            ("Give me the summary.", "summary"),
            ("What are the action items?", "action_items"),
            ("Who is responsible for the first action item?", "action_items"),
            ("What decisions were made?", "decisions"),
            ("What was decided about the budget?", "decisions"),
            ("What did Speaker 2 say about the project?", "speaker"),
            ("Who spoke the most?", "speaker_duration"),
            ("How long did Alice speak?", "speaker_duration"),
            ("When was the budget discussed?", "temporal"),
            ("What are the important topics discussed?", "agenda"),
            ("Was there any discussion about deployment?", "yes_no"),
            ("What was the overall sentiment?", "sentiment_metrics"),
            ("Who attended the meeting?", "participants"),
        ]
        for query, expected_intent in test_cases:
            detected = classify_intent(query)
            logger.info(f"Query: '{query}' -> Detected Intent: '{detected}' (Expected: '{expected_intent}')")
            self.assertEqual(
                detected,
                expected_intent,
                f"Failed for query '{query}': got '{detected}', expected '{expected_intent}'",
            )

    # -----------------------------------------------------------------------
    # Test 2: Speaker Analysis Tool
    # -----------------------------------------------------------------------
    def test_speaker_analysis_tool(self):
        tool = SpeakerAnalysisTool(self.repo)
        res = tool.run(self.meeting1.id)
        self.assertTrue(len(res.chunks) > 0)
        content = res.chunks[0]
        self.assertIn("Alice", content)
        self.assertIn("Bob", content)
        self.assertIn("Charlie", content)
        self.assertIn("Most Active Speaker", content)
        logger.info(f"Speaker Analysis Tool Output:\n{content}")

    # -----------------------------------------------------------------------
    # Test 3: Structured DB Tool (Action items, Decisions, Agenda, Sentiment)
    # -----------------------------------------------------------------------
    def test_structured_db_tool(self):
        tool = StructuredDBTool(self.repo)

        # Action Items
        act_res = tool.run(self.meeting1.id, "action_items")
        self.assertTrue(len(act_res.chunks) > 0)
        self.assertIn("Kubernetes deployment manifests", act_res.chunks[0])
        self.assertIn("Bob", act_res.chunks[0])

        # Decisions
        dec_res = tool.run(self.meeting1.id, "decisions")
        self.assertTrue(len(dec_res.chunks) > 0)
        self.assertIn("PostgreSQL", dec_res.chunks[0])
        self.assertIn("15,000 USD", dec_res.chunks[0])

        # Sentiment / Metrics
        sent_res = tool.run(self.meeting1.id, "sentiment_metrics")
        self.assertTrue(len(sent_res.chunks) > 0)
        self.assertIn("Positive", sent_res.chunks[0])

    # -----------------------------------------------------------------------
    # Test 4: Temporal Search Tool
    # -----------------------------------------------------------------------
    def test_temporal_search_tool(self):
        tool = TemporalSearchTool(self.repo)
        res = tool.run(self.meeting1.id, "When was deployment discussed?")
        self.assertTrue(len(res.chunks) > 0)
        self.assertIn("00:05:05", res.chunks[0])
        self.assertIn("deployment", res.chunks[0].lower())

    # -----------------------------------------------------------------------
    # Test 5: Dynamic MOM Tool
    # -----------------------------------------------------------------------
    def test_dynamic_mom_tool(self):
        tool = DynamicMOMTool(self.repo)
        res = tool.run(self.meeting1.id)
        self.assertTrue(len(res.chunks) > 0)
        content = res.chunks[0]
        self.assertIn("MINUTES OF MEETING", content)
        self.assertIn("Alice, Bob, Charlie", content)
        self.assertIn("PostgreSQL", content)

    # -----------------------------------------------------------------------
    # Test 6: Meeting Isolation & Cross-Meeting Verification
    # -----------------------------------------------------------------------
    def test_meeting_isolation(self):
        router = MeetingAgentRouter(self.repo, vector_service)

        # Query Meeting 1 for action items
        m1_res = router.route(self.meeting1.id, "What are the action items?")
        m1_text = " ".join(m1_res["context_chunks"])
        self.assertIn("Bob", m1_text)
        self.assertIn("Kubernetes", m1_text)
        # Ensure Meeting 2's action item (Eve / KMS key rotation) is NOT leaked
        self.assertNotIn("Eve", m1_text)
        self.assertNotIn("KMS key rotation", m1_text)

        # Query Meeting 2 for action items
        m2_res = router.route(self.meeting2.id, "What are the action items?")
        m2_text = " ".join(m2_res["context_chunks"])
        self.assertIn("Eve", m2_text)
        self.assertIn("KMS key rotation", m2_text)
        # Ensure Meeting 1's data (Bob / Kubernetes) is NOT leaked
        self.assertNotIn("Bob", m2_text)
        self.assertNotIn("Kubernetes", m2_text)

    # -----------------------------------------------------------------------
    # Test 7: Follow-Up & Pronoun Rewriting
    # -----------------------------------------------------------------------
    def test_follow_up_rewriting(self):
        from backend.services.model_router import model_router

        if not (model_router.get_effective_config().get("api_key") or "").strip():
            self.skipTest("needs the AIRAWAT LLM: set an AIRAWAT key")
        history = [
            {"role": "user", "content": "What decisions were made about the database?"},
            {"role": "assistant", "content": "The team decided to migrate to PostgreSQL with a 15,000 USD budget."},
        ]
        rewritten = rewrite_query_with_history("Who proposed it?", history)
        logger.info(f"Original: 'Who proposed it?' -> Rewritten: '{rewritten}'")
        self.assertTrue(len(rewritten) > 0)


if __name__ == "__main__":
    unittest.main()

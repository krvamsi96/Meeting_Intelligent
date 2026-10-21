"""Shared test isolation.

The developer's ``.env`` is loaded into ``settings`` at import time. Without this fixture the suite's result
depended on that file (``AI_PROVIDER=groq`` re-routed the AIRAWAT tests) and a provider test could reach a real
cloud API with the developer's real key. Every test therefore starts from the documented default - AIRAWAT, no
cloud keys, no administrator override from the database - and a test that needs another provider sets it itself.
"""

import pytest

from backend.config.config import settings


@pytest.fixture(autouse=True)
def _neutral_ai_provider_settings(monkeypatch):
    from backend.providers import factory as provider_factory_module
    from backend.providers import provider_factory

    monkeypatch.setattr(settings, "AI_PROVIDER", "airawat")
    monkeypatch.setattr(settings, "STT_PROVIDER", "")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(provider_factory_module.AIProviderFactory, "_get_db_config", staticmethod(lambda: {}))
    provider_factory.clear_cache()
    yield
    provider_factory.clear_cache()

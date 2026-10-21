"""Verify the configured AI providers (text generation AND speech-to-text) from THIS machine; keys are never printed.

    python -m backend.scripts.check_ai_provider            # human readable
    python -m backend.scripts.check_ai_provider --json     # machine readable

AIRAWAT is checked with ``backend.scripts.check_airawat`` (it has its own, more detailed probe). Exit code 0 when
every configured provider answers and its configured model exists, 1 otherwise - usable as a deployment gate.
"""

import argparse
import json
import sys


def _check(name: str) -> dict:
    from backend.providers import provider_factory

    if name == "airawat":
        from backend.services.airawat_health import run_diagnostics

        report = run_diagnostics(deep=False)
        return {"provider": "airawat", "ok": bool(report.get("ok")), "message": report.get("error") or "AIRAWAT answered"}
    provider = provider_factory.get_provider(name)
    res = provider.test_connection()
    return {"provider": name, "ok": bool(res.get("ok")), "message": res.get("message"),
            "model": provider.model, "model_available": res.get("model_available"),
            "transcription_model_available": res.get("transcription_model_available")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()

    from backend.providers import provider_factory

    report: dict = {"ok": True}
    try:
        llm = provider_factory.llm_provider_name()
        report["text_generation"] = _check(llm)
    except Exception as exc:  # noqa: BLE001 - a configuration error is the report
        report["text_generation"] = {"provider": None, "ok": False, "message": str(exc)}
    try:
        stt = provider_factory.stt_provider_name()
        report["speech_to_text"] = (report["text_generation"] if stt == report["text_generation"].get("provider")
                                    else _check(stt))
    except Exception as exc:  # noqa: BLE001
        report["speech_to_text"] = {"provider": None, "ok": False, "message": str(exc)}
    report["ok"] = report["text_generation"]["ok"] and report["speech_to_text"]["ok"]

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["ok"] else 1
    for role in ("text_generation", "speech_to_text"):
        r = report[role]
        print(f"  [{'OK  ' if r['ok'] else 'FAIL'}] {role.replace('_', ' ')}: {r.get('provider') or '-'} - {r.get('message')}")
    print("RESULT:", "AI providers are ready" if report["ok"] else "AI providers need attention")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

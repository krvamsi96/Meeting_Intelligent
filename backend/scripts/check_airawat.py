"""Verify AIRAWAT (the platform's only AI provider) from THIS machine, without exposing the API key.

    python -m backend.scripts.check_airawat            # LLM + Whisper, human readable
    python -m backend.scripts.check_airawat --json     # machine readable
    python -m backend.scripts.check_airawat --llm-only
    python -m backend.scripts.check_airawat --repeat 20 --timeout 20   # measure latency / stall rate from THIS server

Checks: key configured (shown only as a short fingerprint), LLM endpoint + configured model + response format,
Whisper endpoint + request format (file + language, no model) + response format, latency, TLS. Exit code 0 when
everything answers, 1 otherwise - suitable for a deployment gate.
"""

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--llm-only", action="store_true", help="skip the Whisper check")
    parser.add_argument("--timeout", type=float, default=None, help="seconds to wait per request")
    parser.add_argument("--repeat", type=int, default=0, metavar="N",
                        help="send N single-attempt LLM requests and report latency and how many never answered - "
                             "use it to choose AIRAWAT_TIMEOUT_SEC / AIRAWAT_RETRY_COUNT for this server")
    args = parser.parse_args()

    from backend.services.airawat_health import run_diagnostics

    if args.repeat > 0:
        return _measure(args.repeat, args.timeout or 20.0)

    report = run_diagnostics(deep=not args.llm_only, timeout=args.timeout)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("ok") else 1

    def line(ok, label, detail=""):
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")

    print(f"AIRAWAT check  (AI_MODE={report.get('mode')}, AIRAWAT is the only AI provider, "
          f"default language: {report.get('default_language') or 'none - uploads must name it'})")
    line(report.get("configured", False), "API key configured",
         f"fingerprint {report.get('key_fingerprint')}" if report.get("configured") else report.get("error", ""))
    if report.get("configured"):
        llm = report["llm"]
        tries = f", attempt {llm['attempts']}" if llm.get("attempts", 1) > 1 else ""
        line(llm["ok"], f"LLM   {llm.get('model')}", f"{llm.get('latency_ms')} ms{tries}, {llm.get('endpoint')}" if llm["ok"] else llm.get("error", ""))
        if "whisper" in report:
            wh = report["whisper"]
            wtries = f", attempt {wh['attempts']}" if wh.get("attempts", 1) > 1 else ""
            line(wh["ok"], "Whisper (language sent, no model field)",
                 f"{wh.get('latency_ms')} ms{wtries}, {wh.get('endpoint')}" if wh["ok"] else wh.get("error", ""))
        line(bool(report.get("ssl_verify")), "TLS certificate verification", "enabled" if report.get("ssl_verify") else "DISABLED (SSL_VERIFY=false)")
        if report.get("default_language"):
            print(f"  default language configured: {report['default_language']}")
    print("RESULT:", "AIRAWAT is ready" if report.get("ok") else "AIRAWAT is NOT ready - " + str(report.get("error")))
    return 0 if report.get("ok") else 1


def _measure(count: int, timeout: float) -> int:
    """N one-attempt LLM requests: the gateway is known to accept an occasional request and never answer it."""
    import statistics
    import time

    from backend.services.airawat_health import _resolve_config, check_llm

    cfg = _resolve_config()
    if not cfg["api_key"]:
        print("no AIRAWAT API key configured")
        return 1
    answered, stalled, other = [], 0, 0
    for i in range(1, count + 1):
        result = check_llm(cfg, timeout)
        if result["ok"]:
            answered.append(result["latency_ms"])
        elif "timeout" in str(result.get("error", "")):
            stalled += 1
        else:
            other += 1
            print(f"  request {i}: {result.get('error')}")
        time.sleep(0.5)
    print(f"AIRAWAT LLM x{count} (one attempt each, {timeout:.0f}s timeout): answered {len(answered)}, "
          f"no answer within the timeout {stalled}, other errors {other}")
    if answered:
        ordered = sorted(answered)
        print(f"  latency ms: median {int(statistics.median(ordered))}, p90 {ordered[int(0.9 * (len(ordered) - 1))]}, "
              f"max {ordered[-1]}")
    if stalled:
        print(f"  {stalled}/{count} requests were accepted and never answered: keep AIRAWAT_RETRY_COUNT >= 2, and "
              "set AIRAWAT_TIMEOUT_SEC well above the slowest real generation but far below the gateway's own cut-off.")
    return 0 if not other else 1


if __name__ == "__main__":
    sys.exit(main())

"""Datasource diagnosis helper for A-share HTTP fallbacks.

Examples:
    python scripts/diagnose_datasource.py --ticker 688256 --api get_balance_sheet
    python scripts/diagnose_datasource.py --ticker 688256 --all --curr-date 2026-09-20
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from tradingagents.dataflows import a_stock


FINANCIAL_APIS: dict[str, tuple[str, list[tuple[str, Callable[..., Any]]]]] = {
    "get_balance_sheet": (
        "资产负债表",
        [
            ("新浪财经 direct HTTP", a_stock._get_financial_report_sina),
            ("东方财富 datacenter", a_stock._get_financial_report_eastmoney),
            ("腾讯财经", a_stock._get_financial_report_tencent),
        ],
    ),
    "get_cashflow": (
        "现金流量表",
        [
            ("新浪财经 direct HTTP", a_stock._get_financial_report_sina),
            ("东方财富 datacenter", a_stock._get_financial_report_eastmoney),
            ("腾讯财经", a_stock._get_financial_report_tencent),
        ],
    ),
    "get_income_statement": (
        "利润表",
        [
            ("新浪财经 direct HTTP", a_stock._get_financial_report_sina),
            ("东方财富 datacenter", a_stock._get_financial_report_eastmoney),
            ("腾讯财经", a_stock._get_financial_report_tencent),
        ],
    ),
}

SUPPORTED_APIS = [
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "get_industry_comparison",
]


@dataclass
class RequestTrace:
    transport: str
    url: str
    params: dict[str, Any] | None
    headers: dict[str, Any] | None
    elapsed_ms: float
    status_code: int | None
    body_snippet: str
    error: str | None = None


def _sensitive_values_from_env() -> list[str]:
    values = []
    for key, value in os.environ.items():
        upper = key.upper()
        if value and any(token in upper for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            values.append(value)
    return sorted(set(values), key=len, reverse=True)


def _redact_text(value: Any) -> str:
    text = "" if value is None else str(value)
    for secret in _sensitive_values_from_env():
        text = text.replace(secret, "***REDACTED***")
    return text


def _sanitize_mapping(mapping: dict[str, Any] | None) -> dict[str, Any]:
    if not mapping:
        return {}
    out = {}
    for key, value in mapping.items():
        lower = key.lower()
        if any(token in lower for token in ("authorization", "api_key", "apikey", "token", "secret", "password")):
            out[key] = "***REDACTED***"
        else:
            out[key] = _redact_text(value)
    return out


def _safe_json(value: Any) -> str:
    return _redact_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _snippet(text: str, max_lines: int = 12, max_chars: int = 1200) -> str:
    lines = text.splitlines()[:max_lines]
    clipped = "\n".join(lines)
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars] + "... [truncated]"
    return _redact_text(clipped)


def _build_public_url(url: str, params: dict[str, Any] | None) -> str:
    prepared = a_stock._requests.Request("GET", url, params=params).prepare()
    return _redact_text(prepared.url)


@contextlib.contextmanager
def traced_requests() -> list[RequestTrace]:
    traces: list[RequestTrace] = []
    orig_requests_get = a_stock._requests.get
    orig_em_session_get = a_stock._EM_SESSION.get

    def wrap_request(
        transport: str,
        original_get: Callable[..., Any],
        url: str,
        *args,
        **kwargs,
    ):
        params = kwargs.get("params")
        headers = kwargs.get("headers")
        started = time.perf_counter()
        try:
            response = original_get(url, *args, **kwargs)
        except Exception as exc:
            traces.append(
                RequestTrace(
                    transport=transport,
                    url=url,
                    params=params,
                    headers=headers,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    status_code=None,
                    body_snippet="",
                    error=a_stock._exception_summary(exc),
                )
            )
            raise
        traces.append(
            RequestTrace(
                transport=transport,
                url=url,
                params=params,
                headers=headers,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                status_code=getattr(response, "status_code", None),
                body_snippet=_snippet(getattr(response, "text", "")),
            )
        )
        return response

    def wrapped_requests_get(url: str, *args, **kwargs):
        return wrap_request("requests.get", orig_requests_get, url, *args, **kwargs)

    def wrapped_em_get(url: str, *args, **kwargs):
        return wrap_request("eastmoney-session.get", orig_em_session_get, url, *args, **kwargs)

    a_stock._requests.get = wrapped_requests_get
    a_stock._EM_SESSION.get = wrapped_em_get
    try:
        yield traces
    finally:
        a_stock._requests.get = orig_requests_get
        a_stock._EM_SESSION.get = orig_em_session_get


@contextlib.contextmanager
def quiet_dataflow_logger():
    logger = logging.getLogger(a_stock.__name__)
    prev_disabled = logger.disabled
    prev_level = logger.level
    logger.disabled = False
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.disabled = prev_disabled
        logger.setLevel(prev_level)


def _render_trace(trace: RequestTrace) -> str:
    parts = [
        f"- transport: {trace.transport}",
        f"  url: {_build_public_url(trace.url, trace.params)}",
        f"  headers: {_safe_json(_sanitize_mapping(trace.headers))}",
    ]
    if trace.status_code is not None:
        parts.append(f"  status: {trace.status_code}")
    if trace.error:
        parts.append(f"  error: {_redact_text(trace.error)}")
    parts.append(f"  elapsed_ms: {trace.elapsed_ms:.1f}")
    if trace.body_snippet:
        parts.append("  body_snippet:")
        for line in trace.body_snippet.splitlines():
            parts.append(f"    {line}")
    return "\n".join(parts)


def _render_fallback_result(
    source_name: str,
    ok: bool,
    detail: str,
    traces: list[RequestTrace],
) -> str:
    lines = [f"## Fallback source: {source_name}", f"status: {'OK' if ok else 'FAILED'}", f"detail: {_redact_text(detail)}"]
    if traces:
        lines.append("requests:")
        lines.extend(_render_trace(trace) for trace in traces)
    else:
        lines.append("requests: (no HTTP trace captured)")
    return "\n".join(lines)


def diagnose_financial_api(
    api_name: str,
    ticker: str,
    freq: str,
    curr_date: str | None,
) -> str:
    report_type, fetchers = FINANCIAL_APIS[api_name]
    code = a_stock._normalize_ticker(ticker)
    lines = [f"# Diagnose {api_name}", f"ticker: {ticker}", f"normalized_code: {code}", f"report_type: {report_type}", f"freq: {freq}", f"curr_date: {curr_date or '(none)'}"]

    for source_name, fetcher in fetchers:
        with quiet_dataflow_logger(), traced_requests() as traces:
            try:
                df = fetcher(code, report_type, freq, curr_date)
                if df is None or df.empty:
                    detail = "empty dataframe"
                    ok = False
                else:
                    detail = f"rows={len(df)}, columns={list(df.columns)}"
                    ok = True
            except Exception as exc:
                detail = a_stock._exception_summary(exc)
                ok = False
        lines.append("")
        lines.append(_render_fallback_result(source_name, ok, detail, traces))

    with quiet_dataflow_logger(), traced_requests() as traces:
        final_text = getattr(a_stock, api_name)(ticker, freq, curr_date)
    lines.append("")
    lines.append("## Public API result")
    lines.append(_redact_text(final_text[:4000]))
    if traces:
        lines.append("requests:")
        lines.extend(_render_trace(trace) for trace in traces)
    return "\n".join(lines)


def diagnose_industry_comparison(
    ticker: str,
    curr_date: str | None,
    top_n: int,
) -> str:
    lines = [f"# Diagnose get_industry_comparison", f"ticker: {ticker}", f"trade_date: {curr_date or '(none)'}", f"top_n: {top_n}"]
    with quiet_dataflow_logger(), traced_requests() as traces:
        result = a_stock.get_industry_comparison(ticker, curr_date, top_n=top_n)
    lines.append("")
    lines.append("## Public API result")
    lines.append(_redact_text(result[:4000]))
    if traces:
        lines.append("requests:")
        lines.extend(_render_trace(trace) for trace in traces)
    return "\n".join(lines)


def render_diagnosis_report(sections: list[str]) -> str:
    return "\n\n".join(_redact_text(section) for section in sections if section)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True, help="A-share ticker, e.g. 688256")
    parser.add_argument("--api", choices=SUPPORTED_APIS, help="Single API to diagnose")
    parser.add_argument("--all", action="store_true", help="Run all supported diagnoses")
    parser.add_argument("--freq", default="quarterly", choices=["quarterly", "annual"])
    parser.add_argument("--curr-date", default=None, help="YYYY-MM-DD analysis date")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args(argv)
    if not args.all and not args.api:
        parser.error("either --api or --all is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apis = SUPPORTED_APIS if args.all else [args.api]
    sections = []
    for api_name in apis:
        if api_name in FINANCIAL_APIS:
            sections.append(
                diagnose_financial_api(api_name, args.ticker, args.freq, args.curr_date)
            )
        else:
            sections.append(
                diagnose_industry_comparison(args.ticker, args.curr_date, args.top_n)
            )
    print(render_diagnosis_report(sections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

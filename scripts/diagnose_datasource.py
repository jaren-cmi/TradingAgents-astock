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
import re
import time
from dataclasses import dataclass
from io import StringIO
from typing import Any, Callable

import pandas as pd
import requests

from tradingagents.dataflows import a_stock


FINANCIAL_APIS = {
    "get_balance_sheet": "资产负债表",
    "get_cashflow": "现金流量表",
    "get_income_statement": "利润表",
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


def _mask_inline_secret_fields(text: str) -> str:
    patterns = [
        (
            re.compile(
                r'(?i)\b(authorization|api[_-]?key|apikey|token|secret|password)\b'
                r'(\s*[:=]\s*[\'"]?)([^\'"\s,}]+)'
            ),
            r"\1\2***REDACTED***",
        ),
        (
            re.compile(
                r'(?i)([?&](?:authorization|api[_-]?key|apikey|token|secret|password)=)([^&\s]+)'
            ),
            r"\1***REDACTED***",
        ),
    ]
    masked = text
    for pattern, replacement in patterns:
        masked = pattern.sub(replacement, masked)
    return masked


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
    prepared = requests.Request("GET", url, params=params).prepare()
    return _redact_text(prepared.url)


def _run_traced_request(
    transport: str,
    request_fn: Callable[[], Any],
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, Any] | None = None,
) -> tuple[RequestTrace, requests.Response | None, Exception | None]:
    started = time.perf_counter()
    try:
        with quiet_dataflow_logger():
            response = request_fn()
    except Exception as exc:
        return (
            RequestTrace(
                transport=transport,
                url=url,
                params=params,
                headers=headers,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                status_code=None,
                body_snippet="",
                error=a_stock._exception_summary(exc),
            ),
            None,
            exc,
        )
    return (
        RequestTrace(
            transport=transport,
            url=url,
            params=params,
            headers=headers,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            status_code=getattr(response, "status_code", None),
            body_snippet=_snippet(getattr(response, "text", "")),
        ),
        response,
        None,
    )


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
    lines = [
        f"## Fallback source: {source_name}",
        f"status: {'OK' if ok else 'FAILED'}",
        f"detail: {_redact_text(detail)}",
    ]
    if traces:
        lines.append("requests:")
        lines.extend(_render_trace(trace) for trace in traces)
    else:
        lines.append("requests: (no HTTP trace captured)")
    return "\n".join(lines)


def _financial_source_probes(
) -> list[tuple[str, Callable[..., tuple[bool, str, list[RequestTrace]]]]]:
    return [
        ("新浪财经 direct HTTP", _probe_sina_financial),
        ("东方财富 datacenter", _probe_eastmoney_financial),
        ("腾讯财经", _probe_tencent_financial),
    ]


def _probe_sina_financial(
    code: str,
    report_type: str,
    freq: str,
    curr_date: str | None,
) -> tuple[bool, str, list[RequestTrace]]:
    source_type = {
        "资产负债表": "fzb",
        "利润表": "lrb",
        "现金流量表": "llb",
    }[report_type]
    prefix = "sh" if code.startswith("6") else "sz"
    url = "https://quotes.sina.cn/cn/api/openapi.php/CompanyFinanceService.getFinanceReport2022"
    params = {
        "paperCode": f"{prefix}{code}",
        "source": source_type,
        "type": "0",
        "page": "1",
        "num": "20",
    }
    headers = {"User-Agent": a_stock._UA}
    trace, response, exc = _run_traced_request(
        "requests.get",
        lambda: a_stock._http_get(
            url,
            params=params,
            headers=headers,
            timeout=15,
            source_name=f"Sina {report_type} {code}",
        ),
        url,
        params=params,
        headers=headers,
    )
    if exc:
        return False, a_stock._exception_summary(exc), [trace]
    payload = response.json()
    items = payload.get("result", {}).get("data", {}).get(source_type, [])
    df = a_stock._normalize_financial_statement_df(pd.DataFrame(items), report_type)
    df = a_stock._apply_financial_statement_filters(df, freq, curr_date)
    if df.empty:
        return False, "empty dataframe", [trace]
    return True, f"rows={len(df)}, columns={list(df.columns)}", [trace]


def _probe_eastmoney_financial(
    code: str,
    report_type: str,
    freq: str,
    curr_date: str | None,
) -> tuple[bool, str, list[RequestTrace]]:
    url = a_stock._DATACENTER_URL
    params = a_stock._eastmoney_financial_report_params(code, report_type)
    trace, response, exc = _run_traced_request(
        "eastmoney-session.get",
        lambda: a_stock._em_get(url, params=params, timeout=15),
        url,
        params=params,
    )
    if exc:
        return False, a_stock._exception_summary(exc), [trace]
    payload = response.json()
    items = payload.get("result", {}).get("data") or []
    df = a_stock._normalize_financial_statement_df(pd.DataFrame(items), report_type)
    df = a_stock._apply_financial_statement_filters(df, freq, curr_date)
    if df.empty:
        return False, "empty dataframe", [trace]
    return True, f"rows={len(df)}, columns={list(df.columns)}", [trace]


def _probe_tencent_financial(
    code: str,
    report_type: str,
    freq: str,
    curr_date: str | None,
) -> tuple[bool, str, list[RequestTrace]]:
    url_map = {
        "资产负债表": "zcfzb",
        "利润表": "lrfpb",
        "现金流量表": "xjllb",
    }
    url = (
        "https://stock.finance.qq.com/corp1/"
        f"{url_map[report_type]}_detail.php?zq=all&code={a_stock._get_prefix(code)}{code}"
    )
    headers = {"User-Agent": a_stock._UA, "Referer": "https://stock.finance.qq.com/"}
    trace, response, exc = _run_traced_request(
        "requests.get",
        lambda: a_stock._http_get(
            url,
            headers=headers,
            timeout=15,
            source_name=f"Tencent {report_type} {code}",
        ),
        url,
        headers=headers,
    )
    if exc:
        return False, a_stock._exception_summary(exc), [trace]
    tables = pd.read_html(StringIO(response.text))
    for table in tables:
        if table is None or table.empty:
            continue
        work = table.copy()
        if isinstance(work.columns, pd.MultiIndex):
            work.columns = [
                " ".join(str(part).strip() for part in col if str(part).strip())
                for col in work.columns
            ]
        else:
            work.columns = [str(col).strip() for col in work.columns]
        if len(work.columns) < 2:
            continue
        if not any(
            "报" in str(col) or re.search(r"\d{4}-\d{2}-\d{2}", str(col))
            for col in work.columns[1:]
        ):
            continue
        work = a_stock._normalize_financial_statement_df(work, report_type)
        filtered = a_stock._apply_financial_statement_filters(work, freq, curr_date)
        if not filtered.empty:
            return True, f"rows={len(filtered)}, columns={list(filtered.columns)}", [trace]
    return False, "table layout unrecognized or empty dataframe", [trace]


def _probe_industry_comparison(
    ticker: str,
    curr_date: str | None,
    top_n: int,
) -> tuple[str, list[RequestTrace]]:
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1",
        "pz": "100",
        "po": "1",
        "np": "1",
        "fltt": "2",
        "invt": "2",
        "fs": "m:90+t:2",
        "fields": "f2,f3,f4,f12,f13,f14,f104,f105,f128,f136,f140,f141,f207",
    }
    headers = {
        "User-Agent": a_stock._UA,
        "Referer": "https://quote.eastmoney.com/center/boardlist.html#industry_board",
    }
    trace, response, exc = _run_traced_request(
        "eastmoney-session.get",
        lambda: a_stock._em_get(url, params=params, headers=headers, timeout=15),
        url,
        params=params,
        headers=headers,
    )
    if exc:
        return a_stock._exception_summary(exc), [trace]
    payload = response.json()
    items = payload.get("data", {}).get("diff", [])
    if not items:
        return "empty industry list", [trace]
    return f"rows={len(items)}, top_n={top_n}, trade_date={curr_date}", [trace]


def diagnose_financial_api(
    api_name: str,
    ticker: str,
    freq: str,
    curr_date: str | None,
) -> str:
    report_type = FINANCIAL_APIS[api_name]
    code = a_stock._normalize_ticker(ticker)
    lines = [f"# Diagnose {api_name}", f"ticker: {ticker}", f"normalized_code: {code}", f"report_type: {report_type}", f"freq: {freq}", f"curr_date: {curr_date or '(none)'}"]

    for source_name, fetcher in _financial_source_probes():
        try:
            ok, detail, traces = fetcher(code, report_type, freq, curr_date)
        except Exception as exc:
            ok, detail, traces = False, a_stock._exception_summary(exc), []
        lines.append("")
        lines.append(_render_fallback_result(source_name, ok, detail, traces))

    with quiet_dataflow_logger():
        final_text = getattr(a_stock, api_name)(ticker, freq, curr_date)
    lines.append("")
    lines.append("## Public API result")
    lines.append(_redact_text(final_text[:4000]))
    return "\n".join(lines)


def diagnose_industry_comparison(
    ticker: str,
    curr_date: str | None,
    top_n: int,
) -> str:
    lines = [f"# Diagnose get_industry_comparison", f"ticker: {ticker}", f"trade_date: {curr_date or '(none)'}", f"top_n: {top_n}"]
    detail, traces = _probe_industry_comparison(ticker, curr_date, top_n)
    with quiet_dataflow_logger():
        result = a_stock.get_industry_comparison(ticker, curr_date, top_n=top_n)
    lines.append("")
    lines.append("## Probe summary")
    lines.append(_redact_text(detail))
    if traces:
        lines.append("requests:")
        lines.extend(_render_trace(trace) for trace in traces)
    lines.append("")
    lines.append("## Public API result")
    lines.append(_redact_text(result[:4000]))
    return "\n".join(lines)


def render_diagnosis_report(sections: list[str]) -> str:
    return "\n\n".join(
        _mask_inline_secret_fields(_redact_text(section))
        for section in sections
        if section
    )


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

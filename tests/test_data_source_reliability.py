import json
from pathlib import Path

import pandas as pd
import pytest
import requests
from http.client import RemoteDisconnected

from scripts import diagnose_datasource
from tradingagents.agents.quality_gate import _hard_check_report
from tradingagents.dataflows import a_stock


_FIXTURES = Path(__file__).with_name("fixtures")


def _load_fixture(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _report_with_table(*markers: str) -> str:
    body = "基本面分析" * 60
    marker_text = "\n".join(markers)
    return (
        "| 指标 | 值 |\n"
        "| --- | --- |\n"
        f"| 结论 | {body} |\n"
        f"{marker_text}\n"
    )


def test_request_with_retry_retries_timeout(monkeypatch):
    attempts = {"count": 0}
    delays = []

    class _Resp:
        status_code = 200

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise requests.exceptions.Timeout("slow")
        return _Resp()

    monkeypatch.setattr(a_stock.time, "sleep", delays.append)
    resp = a_stock._request_with_retry(flaky, "test source")

    assert attempts["count"] == 3
    assert delays == [0.5, 1.0]
    assert resp.status_code == 200


def test_request_with_retry_does_not_retry_404(monkeypatch):
    attempts = {"count": 0}
    delays = []

    def bad_request():
        attempts["count"] += 1
        exc = requests.exceptions.HTTPError("bad request")
        exc.response = type("Resp", (), {"status_code": 404})()
        raise exc

    monkeypatch.setattr(a_stock.time, "sleep", delays.append)
    with pytest.raises(requests.exceptions.HTTPError):
        a_stock._request_with_retry(bad_request, "test source")

    assert attempts["count"] == 1
    assert delays == []


def test_request_with_retry_does_not_retry_dns_failure(monkeypatch):
    attempts = {"count": 0}
    delays = []

    def dns_broken():
        attempts["count"] += 1
        exc = requests.exceptions.ConnectionError(
            "NameResolutionError: Failed to resolve 'stock.finance.qq.com' "
            "([Errno -2] Name or service not known)"
        )
        raise exc

    monkeypatch.setattr(a_stock.time, "sleep", delays.append)
    with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
        a_stock._request_with_retry(dns_broken, "test source")

    assert attempts["count"] == 1
    assert delays == []
    assert a_stock._classify_source_exception(excinfo.value) == a_stock.FAILURE_DNS_ERROR


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RemoteDisconnected("peer closed"),
        lambda: requests.exceptions.ChunkedEncodingError("chunk broken"),
    ],
)
def test_request_with_retry_retries_remote_disconnect_variants(monkeypatch, factory):
    attempts = {"count": 0}
    delays = []

    class _Resp:
        status_code = 200

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise factory()
        return _Resp()

    monkeypatch.setattr(a_stock.time, "sleep", delays.append)
    resp = a_stock._request_with_retry(flaky, "test source")

    assert attempts["count"] == 3
    assert delays == [0.5, 1.0]
    assert resp.status_code == 200


def test_financial_report_falls_back_to_eastmoney(monkeypatch):
    monkeypatch.setattr(
        a_stock,
        "_get_financial_report_sina",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            requests.exceptions.Timeout("sina timeout")
        ),
    )
    monkeypatch.setattr(
        a_stock,
        "_get_financial_report_eastmoney",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {"报告日": "2026-06-30", "资产总计": 1},
                {"报告日": "2026-12-31", "资产总计": 2},
            ]
        ),
    )
    monkeypatch.setattr(
        a_stock,
        "_get_financial_report_tencent",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should not hit tencent after eastmoney success")
        ),
    )

    out = a_stock.get_balance_sheet("688256", "quarterly", "2026-09-20")

    assert "# Data source: 东方财富 datacenter" in out
    assert "资产总计" in out
    assert "2026-12-31" not in out


def test_financial_report_all_sources_failed_returns_unavailable(monkeypatch):
    monkeypatch.setattr(
        a_stock,
        "_get_financial_report_sina",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            requests.exceptions.Timeout("sina timeout")
        ),
    )
    monkeypatch.setattr(
        a_stock,
        "_get_financial_report_eastmoney",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            requests.exceptions.ConnectionError("eastmoney down")
        ),
    )
    monkeypatch.setattr(
        a_stock,
        "_get_financial_report_tencent",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            requests.exceptions.Timeout("tencent timeout")
        ),
    )

    out = a_stock.get_balance_sheet("688256", "quarterly", "2026-09-20")

    assert "[数据源不可用] 资产负债表" in out
    assert "[确认无数据]" not in out


def test_financial_report_parse_error_is_not_misclassified_as_network(monkeypatch):
    class _Resp:
        status_code = 200
        text = '{"result":{"status":{"code":0},"data":{"report_count":"29"}}}'

        def json(self):
            return {
                "result": {
                    "status": {"code": 0},
                    "data": {
                        "report_count": "29",
                        "report_date": [
                            {
                                "date_value": "20260630",
                                "date_description": "2026半年报",
                                "date_type": 2,
                            }
                        ],
                    },
                }
            }

    monkeypatch.setattr(a_stock, "_http_get", lambda *args, **kwargs: _Resp())
    monkeypatch.setattr(a_stock, "_get_financial_report_eastmoney", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(a_stock, "_get_financial_report_tencent", lambda *args, **kwargs: pd.DataFrame())

    out = a_stock.get_balance_sheet("688256", "quarterly", "2026-09-20")

    assert "PARSE_ERROR" in out
    assert "新浪财经 direct HTTP=PARSE_ERROR" in out
    assert "NETWORK_ERROR" not in out


def test_quality_gate_confirmed_no_data_does_not_reduce_grade():
    grade, detail = _hard_check_report(
        "hot_money",
        _report_with_table(
            "[确认无数据] 龙虎榜：近30日未上龙虎榜。这是事实，可直接写入报告。",
            "[确认无数据] 限售解禁：未来90天无待解禁计划。这是事实，可直接写入报告。",
        ),
    )

    assert grade == "A"
    assert "2 处经确认无数据" in detail


def test_quality_gate_source_failure_still_reduces_grade():
    grade, detail = _hard_check_report(
        "fundamentals",
        _report_with_table(
            "[数据源不可用] 资产负债表：已尝试 新浪/东财/腾讯，均未返回可用数据（最后错误：Timeout）。这是技术故障，不代表该公司没有披露相关数据。",
            "[数据缺失: 因数据源故障未能获取经营性现金流与净利润比值]",
            "[数据缺失: 因数据源故障未能获取行业对比]",
        ),
    )

    assert grade == "C"
    assert "1 处数据源故障" in detail
    assert "2 处未说明原因的数据缺失" in detail


@pytest.fixture
def ths_profit_forecast_html():
    return """
    <html>
      <body>
        <table>
          <thead>
            <tr>
              <th>年度</th><th>预测机构数</th><th>最小值</th><th>均值</th><th>最大值</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>2026</td><td>18</td><td>12.30</td><td>13.50</td><td>14.20</td></tr>
            <tr><td>2027</td><td>16</td><td>15.00</td><td>16.20</td><td>17.80</td></tr>
          </tbody>
        </table>
      </body>
    </html>
    """


def test_profit_forecast_parses_valid_html_without_false_format_error(
    monkeypatch, ths_profit_forecast_html
):
    class _Resp:
        status_code = 200

        def __init__(self, text):
            self.text = text
            self.encoding = None

    monkeypatch.setattr(
        a_stock,
        "_http_get",
        lambda *args, **kwargs: _Resp(ths_profit_forecast_html),
    )
    monkeypatch.setattr(
        a_stock,
        "_tencent_quote",
        lambda codes: {
            codes[0]: {"price": 135.0, "pe_ttm": 80.0},
        },
    )

    out = a_stock.get_profit_forecast("688256", "2026-09-20")

    assert "Error retrieving profit forecast" not in out
    assert "[数据源不可用]" not in out
    assert "FY2026: EPS=13.5" in out
    assert "analysts=18" in out
    assert "Forward PE (FY2026): 10.0x" in out


def test_annual_statement_filter_requires_year_end():
    df = pd.DataFrame(
        [
            {"报告日": "2025-12-15", "资产总计": 1},
            {"报告日": "2025-12-31", "资产总计": 2},
            {"报告日": "2024-12-31", "资产总计": 4},
            {"报告日": "2026-03-31", "资产总计": 3},
        ]
    )

    out = a_stock._apply_financial_statement_filters(df, "annual", "2026-12-31")

    assert out["报告日"].tolist() == ["2025-12-31", "2024-12-31"]


@pytest.mark.parametrize(
    ("report_type", "expected_report_name"),
    [
        ("资产负债表", "RPT_F10_FINANCE_GBALANCE"),
        ("利润表", "RPT_F10_FINANCE_GINCOME"),
        ("现金流量表", "RPT_F10_FINANCE_GCASHFLOW"),
    ],
)
def test_eastmoney_financial_report_params_use_verified_report_names(
    report_type, expected_report_name
):
    params_688 = a_stock._eastmoney_financial_report_params("688256", report_type)
    params_600 = a_stock._eastmoney_financial_report_params("600519", report_type)

    assert params_688["reportName"] == expected_report_name
    assert params_600["reportName"] == expected_report_name
    assert params_688["filter"] == '(SECUCODE="688256.SH")'
    assert params_600["filter"] == '(SECUCODE="600519.SH")'


@pytest.mark.parametrize(
    ("report_type", "sina_row", "eastmoney_row", "expected_columns"),
    [
        (
            "资产负债表",
            {"报告日": "2026-06-30", "资产总计": 10, "负债合计": 4},
            {"REPORT_DATE": "2026-06-30", "TOTAL_ASSETS": 10, "TOTAL_LIABILITIES": 4},
            ["报告日", "资产总计", "负债合计"],
        ),
        (
            "现金流量表",
            {"报告日": "2026-06-30", "经营活动产生的现金流量净额": 3, "筹资活动产生的现金流量净额": 2},
            {"REPORT_DATE": "2026-06-30", "NETCASH_OPERATE": 3, "NETCASH_FINANCE": 2},
            ["报告日", "经营活动产生的现金流量净额", "筹资活动产生的现金流量净额"],
        ),
    ],
)
def test_financial_statement_normalization_aligns_sina_and_eastmoney_fields(
    report_type, sina_row, eastmoney_row, expected_columns
):
    sina = a_stock._normalize_financial_statement_df(pd.DataFrame([sina_row]), report_type)
    eastmoney = a_stock._normalize_financial_statement_df(
        pd.DataFrame([eastmoney_row]), report_type
    )

    assert expected_columns == [col for col in expected_columns if col in sina.columns]
    assert expected_columns == [col for col in expected_columns if col in eastmoney.columns]
    assert sina.loc[0, expected_columns].to_dict() == eastmoney.loc[0, expected_columns].to_dict()


@pytest.mark.parametrize(
    ("report_type", "df", "expected_columns"),
    [
        (
            "资产负债表",
            pd.DataFrame(
                [{" 报告日 ": "2026-06-30", "资产总计": 10, "负债合计": 4}]
            ),
            ["报告日", "资产总计", "负债合计"],
        ),
        (
            "资产负债表",
            pd.DataFrame(
                [{" REPORT_DATE ": "2026-06-30", "TOTAL_ASSETS": 10, "TOTAL_LIABILITIES": 4}]
            ),
            ["报告日", "资产总计", "负债合计"],
        ),
    ],
)
def test_financial_statement_normalization_handles_realistic_dataframe_shapes(
    report_type, df, expected_columns
):
    normalized = a_stock._normalize_financial_statement_df(df, report_type)

    assert normalized.loc[0, expected_columns].to_dict() == {
        "报告日": "2026-06-30",
        "资产总计": 10,
        "负债合计": 4,
    }


@pytest.mark.parametrize(
    ("report_type", "fixture_name", "expected_column", "expected_value"),
    [
        ("资产负债表", "sina_balance_sheet_response.json", "资产总计", "1000.0"),
        ("利润表", "sina_income_statement_response.json", "营业总收入", "210.0"),
        ("现金流量表", "sina_cashflow_response.json", "经营活动产生的现金流量净额", "55.0"),
    ],
)
def test_sina_financial_parser_extracts_real_response_samples(
    report_type, fixture_name, expected_column, expected_value
):
    payload = _load_fixture(fixture_name)

    df = a_stock._parse_sina_financial_report_payload(
        payload,
        {"资产负债表": "fzb", "利润表": "lrb", "现金流量表": "llb"}[report_type],
    )
    normalized = a_stock._normalize_financial_statement_df(df, report_type)

    assert normalized["报告日"].tolist() == ["2026-06-30", "2026-03-31"]
    assert normalized.loc[0, expected_column] == expected_value


def test_eastmoney_financial_report_reaches_http_request_layer(monkeypatch):
    calls = {}

    class _Resp:
        def json(self):
            return {
                "result": {
                    "data": [
                        {
                            "REPORT_DATE": "2026-06-30",
                            "TOTAL_ASSETS": 100,
                            "TOTAL_LIABILITIES": 40,
                        }
                    ]
                }
            }

    def fake_em_get(url, params=None, timeout=None, headers=None):
        calls["url"] = url
        calls["params"] = params
        calls["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(a_stock, "_em_get", fake_em_get)

    df = a_stock._get_financial_report_eastmoney(
        "688256", "资产负债表", "quarterly", "2026-09-20"
    )

    assert calls["url"] == a_stock._DATACENTER_URL
    assert calls["params"]["reportName"] == "RPT_F10_FINANCE_GBALANCE"
    assert not df.empty
    assert df.loc[0, "资产总计"] == 100


def test_diagnosis_report_surfaces_parse_error_category():
    trace = diagnose_datasource.RequestTrace(
        transport="requests.get",
        url="https://example.com",
        params={},
        headers={},
        elapsed_ms=123.4,
        status_code=200,
        body_snippet='{"ok":true}',
    )
    rendered = diagnose_datasource._render_fallback_result(
        "新浪财经 direct HTTP",
        diagnose_datasource.ProbeResult(
            ok=False,
            category=a_stock.FAILURE_PARSE_ERROR,
            detail="empty dataframe after successful parse",
            traces=[trace],
        ),
    )

    assert "category: PARSE_ERROR" in rendered
    assert "status: FAILED" in rendered


def test_diagnosis_report_redacts_env_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")

    rendered = diagnose_datasource.render_diagnosis_report(
        [
            "request url contains sk-secret-value",
            diagnose_datasource._safe_json({"Authorization": "******"}),
        ]
    )

    assert "sk-secret-value" not in rendered
    assert "***REDACTED***" in rendered


def test_diagnosis_report_redacts_inline_secret_fields():
    rendered = diagnose_datasource.render_diagnosis_report(
        [
            "Authorization: inline-secret",
            "https://example.com/path?token=inline-token&ok=1",
        ]
    )

    assert "inline-secret" not in rendered
    assert "inline-token" not in rendered
    assert rendered.count("***REDACTED***") >= 2


def test_diagnosis_parse_args_rejects_api_and_all():
    with pytest.raises(SystemExit):
        diagnose_datasource.parse_args(
            ["--ticker", "688256", "--api", "get_balance_sheet", "--all"]
        )


def test_lockup_empty_results_are_confirmed_no_data(monkeypatch):
    monkeypatch.setattr(a_stock, "_eastmoney_datacenter", lambda *args, **kwargs: [])

    out = a_stock.get_lockup_expiry("688256", "2026-09-20")

    assert "[确认无数据] 限售解禁" in out
    assert "\n无历史解禁记录。" not in out
    assert "\n未来 90 天无待解禁。" not in out

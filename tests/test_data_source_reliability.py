import pandas as pd
import pytest
import requests

from tradingagents.agents.quality_gate import _hard_check_report
from tradingagents.dataflows import a_stock


def _report_with_table(*markers: str) -> str:
    body = "基本面分析" * 60
    marker_text = "\n".join(markers)
    return (
        "| 指标 | 值 |\n"
        "| --- | --- |\n"
        f"| 结论 | {body} |\n"
        f"{marker_text}\n"
    )


def test_request_with_retry_retries_timeout():
    attempts = {"count": 0}

    class _Resp:
        status_code = 200

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise requests.exceptions.Timeout("slow")
        return _Resp()

    resp = a_stock._request_with_retry(flaky, "test source")

    assert attempts["count"] == 3
    assert resp.status_code == 200


def test_request_with_retry_does_not_retry_404():
    attempts = {"count": 0}

    def bad_request():
        attempts["count"] += 1
        exc = requests.exceptions.HTTPError("bad request")
        exc.response = type("Resp", (), {"status_code": 404})()
        raise exc

    with pytest.raises(requests.exceptions.HTTPError):
        a_stock._request_with_retry(bad_request, "test source")

    assert attempts["count"] == 1


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

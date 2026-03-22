# -*- coding: utf-8 -*-
"""Tests for analyzer news prompt hard constraints (Issue #697)."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules["litellm"] = MagicMock()

from src.analyzer import GeminiAnalyzer


class AnalyzerNewsPromptTestCase(unittest.TestCase):
    def test_analysis_prompt_resolves_shared_skill_prompt_state_by_default(self) -> None:
        with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
            analyzer = GeminiAnalyzer()

        fake_state = SimpleNamespace(
            skill_instructions="### 技能 1: 波段低吸\n- 关注支撑确认",
            default_skill_policy="",
        )
        with patch("src.agent.factory.resolve_skill_prompt_state", return_value=fake_state):
            prompt = analyzer._get_analysis_system_prompt("zh", stock_code="600519")

        self.assertIn("### 技能 1: 波段低吸", prompt)
        self.assertNotIn("专注于趋势交易", prompt)

    def test_analysis_prompt_uses_injected_skill_sections_instead_of_hardcoded_trend_baseline(self) -> None:
        with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
            analyzer = GeminiAnalyzer(
                skill_instructions="### 技能 1: 缠论\n- 关注中枢与背驰",
                default_skill_policy="",
            )

        prompt = analyzer._get_analysis_system_prompt("zh", stock_code="600519")

        self.assertIn("### 技能 1: 缠论", prompt)
        self.assertNotIn("专注于趋势交易", prompt)
        self.assertNotIn("多头排列：MA5 > MA10 > MA20", prompt)

    def test_analysis_prompt_keeps_injected_default_policy_for_implicit_default_run(self) -> None:
        with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
            analyzer = GeminiAnalyzer(
                skill_instructions="### 技能 1: 默认多头趋势",
                default_skill_policy="## 默认技能基线（必须严格遵守）\n- **多头排列必须条件**：MA5 > MA10 > MA20",
            )

        prompt = analyzer._get_analysis_system_prompt("zh", stock_code="600519")

        self.assertIn("### 技能 1: 默认多头趋势", prompt)
        self.assertIn("多头排列必须条件", prompt)

    def test_prompt_contains_time_constraints(self) -> None:
        with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
            analyzer = GeminiAnalyzer()

        context = {
            "code": "600519",
            "stock_name": "贵州茅台",
            "date": "2026-03-16",
            "today": {},
            "fundamental_context": {
                "earnings": {
                    "data": {
                        "financial_report": {"report_date": "2025-12-31", "revenue": 1000},
                        "dividend": {"ttm_cash_dividend_per_share": 1.2, "ttm_dividend_yield_pct": 2.4},
                    }
                }
            },
        }
        fake_cfg = SimpleNamespace(
            news_max_age_days=30,
            news_strategy_profile="medium",  # 7 days
        )
        with patch("src.analyzer.get_config", return_value=fake_cfg):
            prompt = analyzer._format_prompt(context, "贵州茅台", news_context="news")

        self.assertIn("近7日的新闻搜索结果", prompt)
        self.assertIn("每一条都必须带具体日期（YYYY-MM-DD）", prompt)
        self.assertIn("超出近7日窗口的新闻一律忽略", prompt)
        self.assertIn("时间未知、无法确定发布日期的新闻一律忽略", prompt)
        self.assertIn("财报与分红（价值投资口径）", prompt)
        self.assertIn("禁止编造", prompt)

    def test_prompt_prefers_context_news_window_days(self) -> None:
        with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
            analyzer = GeminiAnalyzer()

        context = {
            "code": "600519",
            "stock_name": "贵州茅台",
            "date": "2026-03-16",
            "today": {},
            "news_window_days": 1,
        }
        fake_cfg = SimpleNamespace(
            news_max_age_days=30,
            news_strategy_profile="long",  # 30 days if fallback is used
        )
        with patch("src.analyzer.get_config", return_value=fake_cfg):
            prompt = analyzer._format_prompt(context, "贵州茅台", news_context="news")

        self.assertIn("近1日的新闻搜索结果", prompt)
        self.assertIn("超出近1日窗口的新闻一律忽略", prompt)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from copyright_pipeline.prometheus_v2 import parse_prometheus_result_score


class PrometheusV2ParserTest(unittest.TestCase):
    def test_parse_score_success(self):
        score, ok, error = parse_prometheus_result_score("Feedback text. [RESULT] 4")
        self.assertTrue(ok)
        self.assertEqual(score, 4.0)
        self.assertEqual(error, "")

    def test_parse_uses_last_result_token(self):
        text = "bad [RESULT] 9 but final valid [RESULT] 5"
        score, ok, error = parse_prometheus_result_score(text)
        self.assertTrue(ok)
        self.assertEqual(score, 5.0)
        self.assertEqual(error, "")

    def test_parse_failure(self):
        score, ok, error = parse_prometheus_result_score("No result token.")
        self.assertFalse(ok)
        self.assertIsNone(score)
        self.assertEqual(error, "missing_or_invalid_result_token")

    def test_parse_fallback_score_prefix(self):
        score, ok, error = parse_prometheus_result_score("Feedback: short. Score: 3")
        self.assertTrue(ok)
        self.assertEqual(score, 3.0)
        self.assertEqual(error, "")

    def test_parse_fallback_standalone_digit(self):
        score, ok, error = parse_prometheus_result_score("Final answer is 4")
        self.assertTrue(ok)
        self.assertEqual(score, 4.0)
        self.assertEqual(error, "")


if __name__ == "__main__":
    unittest.main()

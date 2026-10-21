"""Tests for Anthropic-powered customer cluster explanations."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from app import get_anthropic_settings
from src.cluster_explanations import (
    generate_cluster_explanations,
)


class ClusterExplanationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profiles = pd.DataFrame(
            [
                {
                    "Cluster": 0,
                    "CustomerCount": 10,
                    "MedianRecencyDays": 5.0,
                    "MedianFrequency": 3,
                    "MedianMonetary": 100.0,
                },
                {
                    "Cluster": 2,
                    "CustomerCount": 8,
                    "MedianRecencyDays": 40.0,
                    "MedianFrequency": 1,
                    "MedianMonetary": 25.0,
                },
            ]
        )

    @patch(
        "app.st.secrets",
        {"anthropic": {"api_key": "configured-key", "model": "configured-model"}},
    )
    def test_reads_credentials_and_model_from_streamlit_secrets(self) -> None:
        self.assertEqual(
            get_anthropic_settings(),
            ("configured-key", "configured-model"),
        )

    @patch("src.cluster_explanations.Anthropic")
    def test_generates_descriptions_for_each_cluster(self, anthropic_client) -> None:
        anthropic_client.return_value.messages.create.return_value = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text='{"0": "Recent, higher-spending customers.", '
                    '"2": "Less recent, lower-spending customers."}',
                )
            ]
        )

        explanations = generate_cluster_explanations(
            self.profiles,
            api_key="test-key",
            model="claude-test-model",
        )

        self.assertEqual(
            explanations,
            {
                0: "Recent, higher-spending customers.",
                2: "Less recent, lower-spending customers.",
            },
        )
        anthropic_client.assert_called_once_with(api_key="test-key", timeout=30.0)
        anthropic_client.return_value.messages.create.assert_called_once()
        self.assertEqual(
            anthropic_client.return_value.messages.create.call_args.kwargs["model"],
            "claude-test-model",
        )

    @patch("src.cluster_explanations.Anthropic")
    def test_rejects_response_missing_a_cluster(self, anthropic_client) -> None:
        anthropic_client.return_value.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"0": "Only one profile."}')]
        )

        with self.assertRaisesRegex(ValueError, "missing cluster 2"):
            generate_cluster_explanations(
                self.profiles,
                api_key="test-key",
                model="claude-test-model",
            )

    @patch("src.cluster_explanations.Anthropic")
    def test_rejects_invalid_json_response(self, anthropic_client) -> None:
        anthropic_client.return_value.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="not-json")]
        )

        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            generate_cluster_explanations(
                self.profiles,
                api_key="test-key",
                model="claude-test-model",
            )


if __name__ == "__main__":
    unittest.main()

"""Generate evidence-based natural-language descriptions of customer clusters."""

import json
from typing import Any

import pandas as pd
from anthropic import Anthropic


PROFILE_COLUMNS = [
    "Cluster",
    "CustomerCount",
    "MedianRecencyDays",
    "MedianFrequency",
    "MedianMonetary",
]


def generate_cluster_explanations(
    profiles: pd.DataFrame,
    api_key: str,
    model: str,
) -> dict[int, str]:
    """Generate one concise explanation for each customer cluster.

    Args:
        profiles: Cluster-level customer counts and median RFM features.
        api_key: Anthropic API key.
        model: Anthropic model identifier.

    Returns:
        A mapping from cluster ID to its generated description.

    Raises:
        ValueError: If the profile data or model response is invalid.
        anthropic.APIError: If the Anthropic API request fails.
    """
    missing_columns = set(PROFILE_COLUMNS).difference(profiles.columns)
    if missing_columns:
        raise ValueError(f"Cluster profiles are missing columns: {sorted(missing_columns)}")
    if profiles.empty:
        return {}

    profile_records = profiles[PROFILE_COLUMNS].to_dict(orient="records")
    cluster_ids = {str(int(record["Cluster"])) for record in profile_records}
    prompt = (
        "Describe the customer segments represented by these cluster profiles. "
        "Use only the supplied counts and medians. Explain recency as days since "
        "the last purchase, frequency as distinct invoices, and monetary as total "
        "spend. Compare clusters where useful, avoid causal claims and unsupported "
        "recommendations, and write one concise business-oriented paragraph per "
        "cluster in English. Return only a valid JSON object whose keys are the "
        "cluster IDs as strings and whose values are the descriptions.\n\n"
        f"Profiles: {json.dumps(profile_records, allow_nan=False)}"
    )

    response = Anthropic(api_key=api_key, timeout=30.0).messages.create(
        model=model,
        max_tokens=min(4096, max(1024, len(profile_records) * 256)),
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()
    if response_text.startswith("```"):
        response_text = response_text.removeprefix("```json").removeprefix("```")
        response_text = response_text.removesuffix("```").strip()

    try:
        parsed: Any = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError("Anthropic returned an invalid JSON response.") from error

    if not isinstance(parsed, dict):
        raise ValueError("Anthropic response must be a JSON object keyed by cluster ID.")

    explanations: dict[int, str] = {}
    for cluster_id in cluster_ids:
        explanation = parsed.get(cluster_id)
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(f"Anthropic response is missing cluster {cluster_id}.")
        explanations[int(cluster_id)] = explanation.strip()

    return explanations

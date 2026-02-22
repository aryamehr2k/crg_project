from __future__ import annotations

from typing import Mapping

DOMAIN_LABELS = {
    "health_risk": "Public Health",
    "climate_risk": "Environmental",
    "education_risk": "Education",
    "policy_risk": "Policy",
}

DRIVER_PHRASES = {
    "pollution": "elevated pollution levels",
    "hospital_access": "limited hospital access",
    "income": "lower income levels",
    "education_level": "lower education attainment",
    "cluster_norm": "regional clustering effects",
}


def _severity(score: float) -> str:
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def map_domain_insight(row: Mapping[str, float | str]) -> tuple[str, str]:
    risks = {
        "health_risk": float(row.get("health_risk", 0.0)),
        "climate_risk": float(row.get("climate_risk", 0.0)),
        "education_risk": float(row.get("education_risk", 0.0)),
        "policy_risk": float(row.get("policy_risk", 0.0)),
    }

    primary_domain = max(risks, key=risks.get)
    score = risks[primary_domain]
    severity = _severity(score)
    domain_label = DOMAIN_LABELS.get(primary_domain, "Regional")

    driver_key = row.get(f"top_driver_{primary_domain}", "")
    driver_phrase = DRIVER_PHRASES.get(str(driver_key), "structural factors")

    if severity == "High":
        action_line = "Immediate mitigation is recommended."
    elif severity == "Medium":
        action_line = "Targeted investment can lower risk quickly."
    else:
        action_line = "Risk remains manageable with routine monitoring."

    explanation = (
        f"{domain_label} risk is {severity.lower()}, driven by {driver_phrase}. {action_line}"
    )

    return explanation, severity

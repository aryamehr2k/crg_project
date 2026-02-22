from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from utils.livedata_client import LiveDataClient

FEATURE_COLUMNS = ["income", "pollution", "education_level", "hospital_access"]
LABEL_COLUMNS = ["health_risk", "climate_risk", "education_risk", "policy_risk"]

RISK_WEIGHT_MAP = {
    "health_risk": [
        ("pollution", 0.5, "direct"),
        ("hospital_access", 0.3, "inverse"),
        ("income", 0.2, "inverse"),
    ],
    "climate_risk": [
        ("pollution", 0.7, "direct"),
        ("education_level", 0.2, "inverse"),
        ("income", 0.1, "inverse"),
    ],
    "education_risk": [
        ("education_level", 0.6, "inverse"),
        ("income", 0.3, "inverse"),
        ("hospital_access", 0.1, "inverse"),
    ],
    "policy_risk": [
        ("income", 0.4, "inverse"),
        ("pollution", 0.4, "direct"),
        ("education_level", 0.2, "inverse"),
    ],
}


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    label_names: list[str]
    scaler: MinMaxScaler
    clusterer: KMeans


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def generate_synthetic_data(num_nodes: int = 100, seed: int = 42) -> pd.DataFrame:
    set_seed(seed)

    income = np.random.normal(55000, 15000, num_nodes).clip(15000, 120000)
    pollution = np.random.normal(55, 20, num_nodes).clip(5, 120)
    education = np.random.normal(0.6, 0.15, num_nodes).clip(0.1, 0.95)
    hospital_access = np.random.normal(0.65, 0.2, num_nodes).clip(0.1, 0.98)

    start_date = datetime.utcnow() - timedelta(days=365 * 2)
    offsets = np.random.randint(0, 365 * 2, size=num_nodes)
    created_at = [start_date + timedelta(days=int(offset)) for offset in offsets]

    counties = [f"County {idx % 8}" for idx in range(num_nodes)]
    localities = [f"District {idx % 20}" for idx in range(num_nodes)]

    df = pd.DataFrame(
        {
            "region_id": np.arange(num_nodes),
            "region_name": [f"Region {idx}" for idx in range(num_nodes)],
            "county": counties,
            "locality": localities,
            "income": income,
            "pollution": pollution,
            "education_level": education,
            "hospital_access": hospital_access,
            "created_at": [dt.isoformat() for dt in created_at],
        }
    )
    return df


def _hash_rng(value: str | None) -> np.random.Generator:
    seed = abs(hash(value or "")) % (2**32)
    return np.random.default_rng(seed)


def _extract_numeric(obj: dict[str, Any] | None, keys: list[str]) -> float | None:
    if not obj:
        return None
    for key in keys:
        value = obj.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _is_current_job(job: dict[str, Any]) -> bool:
    if job.get("isCurrent") or job.get("is_current"):
        return True
    if not job.get("endDate") and not job.get("end_date"):
        return True
    return False


def _pick_primary_job(person: dict[str, Any]) -> dict[str, Any] | None:
    jobs = person.get("jobs") or person.get("positions") or []
    if not isinstance(jobs, list) or not jobs:
        position = person.get("position")
        if isinstance(position, dict):
            return position
        return None
    current = [job for job in jobs if isinstance(job, dict) and _is_current_job(job)]
    if current:
        return current[0]
    for job in jobs:
        if isinstance(job, dict):
            return job
    return None


def _infer_income(person: dict[str, Any]) -> float:
    job = _pick_primary_job(person)
    salary = _extract_numeric(job, ["salary", "compensation", "baseSalary"]) if job else None
    if salary:
        return float(salary)

    title = (job or {}).get("title") or person.get("title") or ""
    title_lower = str(title).lower()
    level = (job or {}).get("level") or person.get("level") or ""
    level_lower = str(level).lower()

    title_map = [
        ("chief", 200000),
        ("cxo", 200000),
        ("vp", 175000),
        ("vice president", 175000),
        ("director", 145000),
        ("principal", 130000),
        ("manager", 115000),
        ("senior", 90000),
        ("lead", 95000),
        ("mid", 70000),
        ("junior", 55000),
        ("entry", 50000),
        ("intern", 35000),
    ]
    for token, value in title_map:
        if token in title_lower:
            return float(value)
    for token, value in title_map:
        if token in level_lower:
            return float(value)

    seniority = (job or {}).get("seniority") or person.get("seniority") or ""
    seniority_lower = str(seniority).lower()
    for token, value in title_map:
        if token in seniority_lower:
            return float(value)

    return 65000.0


def _infer_education_level(person: dict[str, Any]) -> float:
    educations = person.get("education") or person.get("educations") or []
    if not isinstance(educations, list):
        educations = [educations]

    degree_scores = {
        "phd": 0.95,
        "doctor": 0.95,
        "doctorate": 0.95,
        "mba": 0.85,
        "master": 0.85,
        "bachelor": 0.7,
        "associate": 0.6,
        "bootcamp": 0.55,
        "certificate": 0.55,
        "high school": 0.4,
    }

    best = 0.5
    for edu in educations:
        if not isinstance(edu, dict):
            continue
        degree = edu.get("degree") or edu.get("degreeName") or edu.get("level") or ""
        degree_lower = str(degree).lower()
        for key, score in degree_scores.items():
            if key in degree_lower:
                best = max(best, score)

    return float(best)


def _infer_pollution(location: str | None) -> float:
    rng = _hash_rng(location)
    value = rng.normal(55, 18)
    return float(np.clip(value, 5, 120))


def _infer_hospital_access(location: str | None) -> float:
    rng = _hash_rng(location)
    value = rng.normal(0.65, 0.18)
    return float(np.clip(value, 0.1, 0.98))


def _extract_location(person: dict[str, Any]) -> str | None:
    details = person.get("location_details")
    if isinstance(details, dict):
        parts = [details.get("locality"), details.get("region"), details.get("country")]
        location = ", ".join([part for part in parts if part])
        if location:
            return location
    for key in ["location", "region", "city", "state", "country"]:
        value = person.get(key)
        if isinstance(value, str) and value.strip():
            return value
    job = _pick_primary_job(person)
    if job:
        details = job.get("location_details")
        if isinstance(details, dict):
            parts = [details.get("locality"), details.get("region"), details.get("country")]
            location = ", ".join([part for part in parts if part])
            if location:
                return location
        for key in ["location", "city", "state", "country"]:
            value = job.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _build_region_name(person: dict[str, Any]) -> str | None:
    details = person.get("location_details")
    if isinstance(details, dict):
        parts = [details.get("locality"), details.get("region"), details.get("country")]
        label = ", ".join([part for part in parts if part])
        if label:
            return label
        raw = details.get("raw")
        if isinstance(raw, str) and raw.strip():
            return raw

    location = person.get("location")
    if isinstance(location, str) and location.strip():
        return location

    country = person.get("country")
    if isinstance(country, str) and country.strip():
        return country

    return None


def _looks_like_person(item: dict[str, Any]) -> bool:
    keys = set(item.keys())
    return bool({"firstName", "lastName", "fullName", "name", "linkedinUrl"} & keys)


def _extract_people(payload: Any) -> list[dict[str, Any]]:
    people: list[dict[str, Any]] = []

    def walk(item: Any, depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(item, dict):
            if "person" in item and isinstance(item["person"], dict):
                people.append(item["person"])
            if _looks_like_person(item):
                people.append(item)
            for key in ("people", "results", "matches", "data", "items", "hits"):
                value = item.get(key)
                if isinstance(value, list):
                    for sub in value:
                        walk(sub, depth + 1)
        elif isinstance(item, list):
            for sub in item:
                walk(sub, depth + 1)

    walk(payload)

    deduped = []
    seen = set()
    for person in people:
        person_id = person.get("id") or person.get("personId") or person.get("linkedinUrl")
        key = person_id or (person.get("firstName"), person.get("lastName"), person.get("name"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(person)
    return deduped


def _people_to_dataframe(people: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for idx, person in enumerate(people):
        location = _extract_location(person)
        region_name = _build_region_name(person) or location or "Unknown"
        job = _pick_primary_job(person)
        details = person.get("location_details")
        if not isinstance(details, dict) and job:
            details = job.get("location_details")
        if not isinstance(details, dict):
            details = {}
        county = details.get("county")
        locality = details.get("locality") or details.get("region")
        if not locality and isinstance(region_name, str):
            locality = region_name.split(",")[0].strip()
        if not county and isinstance(region_name, str):
            for part in region_name.split(","):
                if "county" in part.lower():
                    county = part.strip()
                    break
        income = _infer_income(person)
        education_level = _infer_education_level(person)
        pollution = _infer_pollution(location)
        hospital_access = _infer_hospital_access(location)
        region_id = person.get("id") or person.get("personId") or idx
        created_at = (
            person.get("created_at")
            or person.get("info_change_detected_at")
            or person.get("title_change_detected_at")
            or person.get("company_change_detected_at")
        )

        records.append(
            {
                "region_id": region_id,
                "region_name": region_name,
                "county": county,
                "locality": locality,
                "income": income,
                "pollution": pollution,
                "education_level": education_level,
                "hospital_access": hospital_access,
                "created_at": created_at,
            }
        )
    return pd.DataFrame(records)


def load_livedata_dataframe(data_dir: Path) -> pd.DataFrame | None:
    _maybe_load_dotenv()
    search_path = data_dir / "livedata_search.json"
    find_path = data_dir / "livedata_find.json"

    if not search_path.exists() and not find_path.exists():
        raise FileNotFoundError("No Live Data query file found in data/raw")

    client = LiveDataClient.from_env()
    if search_path.exists():
        query = json.loads(search_path.read_text())
        response = client.search_people(query)
    else:
        query = json.loads(find_path.read_text())
        response = client.find_people(query)

    people = _extract_people(response)
    if not people:
        return None
    return _people_to_dataframe(people)


def load_livedata_json_file(path: Path) -> pd.DataFrame | None:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        people = [item for item in data if isinstance(item, dict)]
    else:
        people = _extract_people(data)
    if not people:
        return None
    return _people_to_dataframe(people)


def _maybe_load_livedata_json(data_dir: Path) -> pd.DataFrame | None:
    env_path = os.environ.get("LIVEDATA_JSON_PATH")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates += [
        data_dir.parent / "live_data_persons_history_combined.json",
        data_dir.parent / "livedata_people.json",
        data_dir / "live_data_persons_history_combined.json",
        data_dir / "livedata_people.json",
        Path("/home/kay/Downloads/live_data_persons_history_combined.json"),
    ]

    for path in candidates:
        if path and path.exists():
            return load_livedata_json_file(path)
    return None


def normalize_features(df: pd.DataFrame, scaler: MinMaxScaler | None = None) -> tuple[np.ndarray, MinMaxScaler]:
    scaler = scaler or MinMaxScaler()
    values = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if not hasattr(scaler, "n_features_in_"):
        normalized = scaler.fit_transform(values)
    else:
        normalized = scaler.transform(values)
    return normalized, scaler


def compute_clusters(features: np.ndarray, n_clusters: int = 3, clusterer: KMeans | None = None) -> tuple[np.ndarray, KMeans]:
    clusterer = clusterer or KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    if not hasattr(clusterer, "cluster_centers_"):
        labels = clusterer.fit_predict(features)
    else:
        labels = clusterer.predict(features)
    return labels, clusterer


def build_risk_scores(norm_features: np.ndarray) -> np.ndarray:
    income = norm_features[:, FEATURE_COLUMNS.index("income")]
    pollution = norm_features[:, FEATURE_COLUMNS.index("pollution")]
    education = norm_features[:, FEATURE_COLUMNS.index("education_level")]
    hospital = norm_features[:, FEATURE_COLUMNS.index("hospital_access")]

    health = 0.5 * pollution + 0.3 * (1 - hospital) + 0.2 * (1 - income)
    climate = 0.7 * pollution + 0.2 * (1 - education) + 0.1 * (1 - income)
    education_risk = 0.6 * (1 - education) + 0.3 * (1 - income) + 0.1 * (1 - hospital)
    policy = 0.4 * (1 - income) + 0.4 * pollution + 0.2 * (1 - education)

    scores = np.vstack([health, climate, education_risk, policy]).T
    scores = np.clip(scores, 0.0, 1.0)
    return scores.astype(np.float32)


def add_explanations(df: pd.DataFrame, norm_features: np.ndarray) -> pd.DataFrame:
    feature_index = {name: idx for idx, name in enumerate(FEATURE_COLUMNS)}

    for label in LABEL_COLUMNS:
        contributions = []
        for feature, weight, direction in RISK_WEIGHT_MAP[label]:
            values = norm_features[:, feature_index[feature]]
            if direction == "inverse":
                values = 1 - values
            contributions.append(weight * values)
        contributions = np.vstack(contributions).T
        top_indices = np.argmax(contributions, axis=1)
        top_features = [RISK_WEIGHT_MAP[label][idx][0] for idx in top_indices]
        df[f"top_driver_{label}"] = top_features

    return df


def prepare_dataset(
    data_dir: Path,
    processed_dir: Path,
    num_nodes: int = 100,
    seed: int = 42,
) -> DatasetBundle:
    data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = data_dir / "nodes.csv"
    processed_path = processed_dir / "nodes.csv"

    _maybe_load_dotenv()
    use_api = os.environ.get("USE_LIVEDATA_API", "").lower() in {"1", "true", "yes"}
    use_api = use_api or os.environ.get("LIVEDATA_USE_API", "").lower() in {"1", "true", "yes"}
    use_json = os.environ.get("USE_LIVEDATA_JSON", "").lower() in {"1", "true", "yes"}
    use_json = use_json or bool(os.environ.get("LIVEDATA_JSON_PATH"))
    use_json = use_json or (data_dir.parent / "live_data_persons_history_combined.json").exists()
    use_json = use_json or (data_dir.parent / "livedata_people.json").exists()
    use_json = use_json or (data_dir / "live_data_persons_history_combined.json").exists()
    use_json = use_json or (data_dir / "livedata_people.json").exists()
    max_nodes = int(os.environ.get("LIVEDATA_MAX_NODES", "1000"))

    df = None
    if use_json:
        try:
            df = _maybe_load_livedata_json(data_dir)
        except Exception as exc:
            print(f"Live Data JSON unavailable, falling back to API/synthetic data: {exc}")

    if df is None and use_api:
        try:
            df = load_livedata_dataframe(data_dir)
            if df is not None and len(df) < 2:
                df = None
        except Exception as exc:
            print(f"Live Data API unavailable, falling back to synthetic data: {exc}")

    if df is None:
        if processed_path.exists():
            df = pd.read_csv(processed_path)
        else:
            df = generate_synthetic_data(num_nodes=num_nodes, seed=seed)
            df.to_csv(raw_path, index=False)

    if max_nodes > 0 and len(df) > max_nodes:
        df = df.sample(n=max_nodes, random_state=seed).reset_index(drop=True)

    norm_features, scaler = normalize_features(df)
    clusters, clusterer = compute_clusters(norm_features)

    df = df.copy()
    for idx, feature in enumerate(FEATURE_COLUMNS):
        df[f"{feature}_norm"] = norm_features[:, idx]
    df["cluster"] = clusters

    risk_scores = build_risk_scores(norm_features)
    for idx, label in enumerate(LABEL_COLUMNS):
        df[label] = risk_scores[:, idx]

    df = add_explanations(df, norm_features)

    df.to_csv(processed_path, index=False)

    cluster_norm = clusters.astype(np.float32) / max(1, (clusters.max() if len(clusters) else 1))
    features = np.hstack([norm_features, cluster_norm.reshape(-1, 1)])
    feature_names = FEATURE_COLUMNS + ["cluster_norm"]

    return DatasetBundle(
        df=df,
        features=features.astype(np.float32),
        labels=risk_scores.astype(np.float32),
        feature_names=feature_names,
        label_names=LABEL_COLUMNS,
        scaler=scaler,
        clusterer=clusterer,
    )


def save_artifacts(bundle: DatasetBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scaler.pkl").open("wb") as handle:
        pickle.dump(bundle.scaler, handle)
    with (output_dir / "clusterer.pkl").open("wb") as handle:
        pickle.dump(bundle.clusterer, handle)

    metadata = {
        "feature_names": bundle.feature_names,
        "label_names": bundle.label_names,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

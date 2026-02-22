from __future__ import annotations

from pathlib import Path

from models.predict import predict_and_save
from models.train import train_model


def run_pipeline() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "outputs"
    model_path = base_dir / "models" / "model.pt"
    predictions_path = output_dir / "predictions.csv"

    if not model_path.exists():
        train_model(
            data_dir=data_dir,
            processed_dir=processed_dir,
            output_dir=output_dir,
            model_path=model_path,
        )

    df = predict_and_save(
        data_dir=data_dir,
        processed_dir=processed_dir,
        model_path=model_path,
        output_path=predictions_path,
    )

    print("Prediction sample:")
    preview_cols = ["region_id"]
    if "region_name" in df.columns:
        preview_cols.append("region_name")
    preview_cols += ["final_score"] + [col for col in df.columns if col.endswith("_risk")]
    preview_cols = [col for col in preview_cols if col in df.columns]
    print(df[preview_cols].head())

    print("\nSummary checks:")
    print(f"Total regions: {len(df)}")
    if "health_risk" in df.columns:
        print(f"Average health risk: {df['health_risk'].mean():.3f}")
    if "region_name" in df.columns:
        ca_mask = df["region_name"].str.contains("California", case=False, na=False)
        if ca_mask.any() and (~ca_mask).any():
            ca_avg = df.loc[ca_mask, "health_risk"].mean()
            rest_avg = df.loc[~ca_mask, "health_risk"].mean()
            print(f"California avg health risk: {ca_avg:.3f}")
            print(f"Rest of US avg health risk: {rest_avg:.3f}")
    if "final_score" in df.columns:
        top = df.sort_values("final_score", ascending=False).head(5)
        cols = [col for col in ["region_id", "region_name", "final_score"] if col in top.columns]
        print("Top 5 regions by final score:")
        print(top[cols])
    if "top_driver_health_risk" in df.columns:
        drivers = df["top_driver_health_risk"].value_counts().head(3)
        print("Top health risk drivers (count):")
        print(drivers)


if __name__ == "__main__":
    run_pipeline()

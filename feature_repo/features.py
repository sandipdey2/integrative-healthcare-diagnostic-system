"""Feast definitions for the IHDS clinical features.

Source: data/patient_features.parquet, written by `python -m ihds.mlops.prepare` from data/pima_diabetes.csv
(zeros already replaced by missing values). patient_id is the CSV row number and event_timestamp is one fixed
date from params.yaml; the dataset has neither real IDs nor timestamps.
"""
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float64, Int64

HERE = Path(__file__).resolve().parent

patient = Entity(name="patient", join_keys=["patient_id"], description="Row number in data/pima_diabetes.csv")

clinical_source = FileSource(
    name="pima_clinical_source",
    path=str(HERE / "data" / "patient_features.parquet"),
    timestamp_field="event_timestamp",
)

clinical_features = FeatureView(
    name="patient_clinical_features",
    entities=[patient],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="Pregnancies", dtype=Float64),
        Field(name="Glucose", dtype=Float64),
        Field(name="BloodPressure", dtype=Float64),
        Field(name="SkinThickness", dtype=Float64),
        Field(name="Insulin", dtype=Float64),
        Field(name="BMI", dtype=Float64),
        Field(name="DiabetesPedigreeFunction", dtype=Float64),
        Field(name="Age", dtype=Float64),
        Field(name="Outcome", dtype=Int64),
    ],
    online=True,
    source=clinical_source,
    tags={"source": "pima_diabetes.csv"},
)

risk_model_features = FeatureService(
    name="risk_model_v1",
    features=[clinical_features[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                                 "DiabetesPedigreeFunction", "Age"]]],
    description="The 8 inputs of the IHDS risk model (no label).",
)

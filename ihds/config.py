"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "pima_diabetes.csv"
KB_PATH = ROOT / "data" / "clinical_knowledge.csv"
REPORTS_DIR = ROOT / "monitoring" / "reports"
ARTIFACTS_DIR = ROOT / "artifacts"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

# Zeros in these columns are physiologically impossible -> treated as missing.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Plausible input ranges used by the input guard (inclusive).
VALID_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (40, 400),
    "BloodPressure": (30, 150),
    "SkinThickness": (5, 100),
    "Insulin": (10, 900),
    "BMI": (12, 70),
    "DiabetesPedigreeFunction": (0.05, 2.5),
    "Age": (21, 100),
}
# Columns stored as whole numbers.
INTEGER_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "Age"]

SEED = 42
TEST_SIZE = 0.2

# Risk bands on the model probability.
LOW_MAX = 0.30
HIGH_MIN = 0.60
BORDERLINE = (0.40, 0.60)

LLM_MODEL = "gpt-4o-mini"

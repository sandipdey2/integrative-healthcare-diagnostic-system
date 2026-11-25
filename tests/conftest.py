import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ihds import data, model  # noqa: E402
from ihds.config import FEATURES  # noqa: E402


@pytest.fixture(scope="session")
def split():
    return data.split(data.load_clean())


@pytest.fixture(scope="session")
def pipe(split):
    return model.train(split[0])


@pytest.fixture(scope="session")
def imputed_train(split, pipe):
    return pd.DataFrame(pipe.named_steps["impute"].transform(split[0][FEATURES]), columns=FEATURES)


@pytest.fixture
def good_record():
    return {"Pregnancies": 2, "Glucose": 150, "BloodPressure": 72, "SkinThickness": 30, "Insulin": 120,
            "BMI": 34.0, "DiabetesPedigreeFunction": 0.5, "Age": 45}


@pytest.fixture(scope="session")
def system():
    from ihds.pipeline import build_system

    return build_system(gan_steps=800)


@pytest.fixture(autouse=True)
def isolate_audit_log(tmp_path, monkeypatch, request):
    """Point the audit log at a per-test SQLite file so tests never touch Neon or data/audit.db.

    Tests marked `postgres` keep the caller's DATABASE_URL (the live integration test).
    """
    from ihds import audit_log

    if request.node.get_closest_marker("postgres"):
        audit_log.reset()
        yield
        audit_log.reset()
        return
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("IHDS_AUDIT_SQLITE", str(tmp_path / "audit.db"))
    audit_log.reset()
    yield
    audit_log.reset()


import json

from ihds import monitoring


def test_reports_generated(tmp_path, pipe, split, imputed_train):
    train, test = split
    synthetic = imputed_train.sample(200, random_state=0)
    res = monitoring.run_all(pipe, train, test, synthetic, out_dir=tmp_path)
    for name in ("holdout", "shifted", "synthetic"):
        assert (tmp_path / f"{name}_report.html").stat().st_size > 10_000
        json.loads((tmp_path / f"{name}_report.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["holdout"]["drifted_feature_names"] == []
    assert summary["shifted"]["drifted_feature_names"] == ["Glucose"]
    assert summary["shifted"]["precision"] < summary["holdout"]["precision"]
    assert 0.6 < summary["holdout"]["accuracy"] < 0.9
    assert res["synthetic"]["current_rows"] == 200

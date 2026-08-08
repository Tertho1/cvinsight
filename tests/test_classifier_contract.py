"""Contract tests for the exported quality classifiers (results/classifier_v2_*.pkl).

Mirrors the exact load/predict/predict_proba contract used by app/app.py:

  model = joblib.load(...)
  raw   = model.predict([text])[0]                 # -> str label
  classes = model.classes_ if present else model.label_classes_
  if isinstance(raw, (int, float, np.integer, np.floating, None)): label = classes[int(raw)]
  else: label = str(raw)
  proba = model.predict_proba([text])[0]           # -> len-3, sums ~1, order == classes_

Also tests edge inputs the app may hit: empty / short / long / garbage / noisy text.
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_best_classifier import CLASSES, QualityPipeline  # noqa: E402

ARTIFACTS = [
    ROOT / "results" / "classifier_v2_rf_sm_2026_08_08.pkl",
    ROOT / "results" / "classifier_v2_xgb_sm_merged_2026_08_08.pkl",
]
DEPLOYED = ROOT / "models" / "xgb_classifier.pkl"
TEXTS = {
    "empty": "",
    "short": "John Doe",
    "long": "Summary\nSenior software engineer with 10 years of experience building "
            "distributed systems in Python, Go and Kubernetes. Led teams of 8, "
            "architected microservices handling 1M RPS, cut latency 40%. "
            "AWS Certified Solutions Architect. " * 50,
    "garbage": "\x00\x01\x02\ufffd\ufffd~~~!!!@@@###\n\n\t",
    "noisy": "Python Python python PYTHON 12345 12345 $%@# senior engineer "
             "engineer engineer engineer developer developer",
    "normal": "Project Manager at AT&T | 2013-01-01 - 2024-03-20\nManaged "
              "vendors, tracked metrics, ensured compliance. Bachelor Commerce "
              "at Mumbai University. Skills: Python, Selenium, AWS, Docker.",
}


def _classify_text(model, text):
    raw = model.predict([text])[0]
    classes = None
    if hasattr(model, "label_classes_"):
        classes = list(model.label_classes_)
    elif hasattr(model, "classes_"):
        classes = list(model.classes_)
    if classes is not None:
        classes = [str(c) for c in classes]
    if isinstance(raw, (int, float, np.integer, np.floating, type(None))):
        if classes is not None:
            label = classes[int(raw)]
        else:
            label = ["Average", "Strong", "Weak"][int(raw)]
    else:
        label = str(raw)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
    return label, proba, classes


@pytest.fixture(params=ARTIFACTS, ids=[p.stem for p in ARTIFACTS])
def model(request):
    if not request.param.exists():
        pytest.skip(f"{request.param.name} not present")
    return joblib.load(str(request.param))


def test_classes_and_label_classes(model):
    assert hasattr(model, "classes_")
    assert hasattr(model, "label_classes_")
    classes = list(model.classes_)
    assert [str(c) for c in classes] == CLASSES, classes


def test_predict_returns_string_label(model):
    for name, text in TEXTS.items():
        label, _, _ = _classify_text(model, text)
        assert isinstance(label, str), (name, label)
        assert label in CLASSES, (name, label)


def test_predict_proba_shape_and_sums(model):
    for name, text in TEXTS.items():
        _, proba, _ = _classify_text(model, text)
        assert proba is not None
        proba = np.asarray(proba).ravel()
        assert proba.shape == (3,), (name, proba.shape)
        assert np.all(proba >= -1e-9), (name, proba)
        assert abs(proba.sum() - 1.0) < 1e-6, (name, proba.sum())


def test_proba_matches_predict_argmax(model):
    for name, text in TEXTS.items():
        label, proba, _ = _classify_text(model, text)
        idx = CLASSES.index(label)
        assert CLASSES[int(np.argmax(proba))] == label, (name, label, proba)


def test_predict_proba_rows_stable_batch(model):
    preds = model.predict(list(TEXTS.values()))
    probas = np.asarray(model.predict_proba(list(TEXTS.values())))
    assert probas.shape == (len(TEXTS), 3)
    for i, p in enumerate(probas):
        assert abs(p.sum() - 1) < 1e-6
    for i, label in enumerate(preds):
        assert CLASSES.index(str(label)) == int(np.argmax(probas[i]))


def test_app_contract_end_to_end(model):
    for name, text in TEXTS.items():
        label, proba, classes = _classify_text(model, text)
        assert label in CLASSES
        assert classes == CLASSES
        assert proba is not None
        assert len(proba) == 3


@pytest.fixture()
def deployed_model():
    if not DEPLOYED.exists():
        pytest.skip(f"{DEPLOYED.name} not present")
    return joblib.load(str(DEPLOYED))


def test_deployed_model_renders_labels_not_indices(deployed_model):
    # Regression: the deployed xgb_classifier.pkl has INTEGER classes_ but string
    # label_classes_. classify_text app prefers label_classes_ so the app renders
    # "Average"/"Strong"/"Weak" instead of "0"/"1"/"2".
    for name, text in TEXTS.items():
        label, proba, classes = _classify_text(deployed_model, text)
        assert label in CLASSES, (name, label)
        assert proba is not None and len(np.asarray(proba).ravel()) == 3, name
        assert set(str(c) for c in classes) == set(CLASSES), (name, classes)


def test_deployed_label_classes_cover_all_int_raws(deployed_model):
    labels = list(deployed_model.label_classes_)
    assert len(labels) == len(deployed_model.classes_)
    for text in TEXTS.values():
        raw = deployed_model.predict([text])[0]
        assert 0 <= int(raw) < len(labels), (repr(raw), labels)

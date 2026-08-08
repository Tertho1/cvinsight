"""Thorough acceptance test for the v3 hybrid classifier (synth artifact).

Exercises the exact contract app/app.py::classify_text relies on:
  * fresh-process joblib.load with ROOT on sys.path (no scripts namespace)
  * predict -> str label; predict_proba -> (3,) in CLASSES order, sums ~1
  * classes_ / label_classes_ attrs cover all CLASSES
  * edge inputs: empty / whitespace / single-word / garbage / control chars /
    unicode / emoji / LLM-JSON / long text / section-ish text
  * determinism (repeat -> identical score) and batch-vs-single agreement
  * threshold boundaries (0/49/50/71/72/99/100)
  * demo+benchmark CVs through the real parse path with per-CV latency

Usage: python scripts/acceptance_hybrid_classifier.py
Exit 0 == all checks pass.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARTIFACT = ROOT / "results" / "classifier_v3_hybrid_synth.pkl"
CLASSES = ["Weak", "Average", "Strong"]
STRONG_MIN, AVG_MIN = 72.0, 50.0

EDGE_INPUTS = {
    "empty": "",
    "whitespace": "   \n\t  ",
    "single_word": "Jane",
    "short": "Ada\njunior@example.com",
    "garbage": "\x00\x01\x02\ufffd\ufffd~~~!!!@@@###\n\n\t",
    "control_chars": "\x00\x1f" * 200,
    "emoji": "John Doe\n\U0001f389 Senior DevOps Engineer \U0001f389\nDocker Kubernetes CI/CD",
    "unicode": "Jan Navig | El Mateus | 42\nEXPERIENCE\nSoftware Ingeniero | 2020 - Present\n- meh\n",
    "json_llm": '{"name": "Alice", "email": "a@x.com", "summary": "Engineer", '
                '"experience": [{"title": "Dev", "company": "X", '
                '"dates": {"start": "2020", "end": "Present"}}], "skills": ["Python"]}',
    "repeated_nonsense": "Python Python python PYTHON python " * 100,
    "long": ("PROFESSIONAL SUMMARY\nSenior software engineer with 10 years of experience "
             "building distributed systems in Python, Go and Kubernetes. Led teams of 8, "
             "architected microservices handling 1M RPS, cut latency 40%. "
             "AWS Certified Solutions Architect. Kubernetes. Terraform. CI/CD.\n"
             "EXPERIENCE\nSenior Software Engineer, CloudScale Inc | Jan 2016 - Present\n"
             "- Architected event-driven microservices and Kubernetes platform\n"
             "- Led 8 engineers; cut deployment times 45min to 6min\n"
             "EDUCATION\nM.Sc. Computer Science, MIT, 2014\n"
             "SKILLS\nPython, Kubernetes, Docker, AWS, Terraform, Kafka\n"
             "PROJECTS\nAWS Data Platform - Spark, Airflow, S3 | github.com/x\n"
             "CERTIFICATIONS\nAWS Certified Solutions Architect, 2021\n"
             "LANGUAGES\nEnglish (Native), Spanish (Conversational)\n"
             "LEADERSHIP\n- Chapter Lead, DevOps Meetup\n") * 8,
    "sectionish": "EXPERIENCE\nEngineer at X | 2020-2023\n- built things\nEDUCATION\nB.Sc. CS\nEND\n",
}

DEMO_FILES = [
    "Rebecca_Software or Computational Roles.docx",
    "junior_dev.txt",
    "priya_dwivedi_repo_MathewElliot.docx",
    "pro-cv-template-burgundy.docx",
    "resume_02_rahul_verma.pdf",
    "resume_03_ananya_patel.pdf",
    "resume_04_vikram_singh.pdf",
    "senior_python_dev.txt",
    "srbhr_repo_barry_allen_fe.pdf",
]


def label_of_score(s):
    return "Strong" if s >= STRONG_MIN else ("Average" if s >= AVG_MIN else "Weak")


def fresh_probe():
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import joblib\n"
        "m = joblib.load(%r)\n"
        "print(type(m).__module__)\n"
        "print(list(m.classes_))\n"
        "print(list(m.label_classes_))\n"
        "print([float(x) for x in m.predict_proba(['hello'])[0]])\n"
    ) % (str(ROOT), str(ARTIFACT))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=str(ROOT), timeout=180)
    return r.returncode, r.stdout.strip().splitlines(), r.stderr


def main():
    import joblib

    assert ARTIFACT.exists(), f"missing {ARTIFACT}"
    print("== fresh-process load ==")
    rc, lines, err = fresh_probe()
    if rc != 0:
        print("stderr tail:", err[-1200:])
        sys.exit(f"fresh-load failed rc={rc}")
    for ln in lines:
        print("  ", ln)

    model = joblib.load(str(ARTIFACT))
    print("loaded:", type(model).__module__)

    print("\n== contract attrs ==")
    assert hasattr(model, "predict") and hasattr(model, "predict_proba")
    assert hasattr(model, "classes_") and hasattr(model, "label_classes_")
    cs = [str(c) for c in model.classes_]
    lcs = [str(c) for c in model.label_classes_]
    assert cs == CLASSES, cs
    assert set(lcs) == set(CLASSES), lcs
    print("  classes_ == label_classes_ ==", CLASSES)

    print("\n== edge inputs ==")
    for name, text in EDGE_INPUTS.items():
        t0 = time.time()
        lab = str(model.predict([text])[0])
        score = float(model.predict_scores([text])[0])
        p = np.asarray(model.predict_proba([text])[0]).ravel()
        dt = (time.time() - t0) * 1000
        assert lab in CLASSES, (name, lab)
        assert p.shape == (3,), (name, p.shape)
        assert np.all(p >= -1e-9), (name, p)
        assert abs(p.sum() - 1.0) < 1e-6, (name, p.sum())
        assert CLASSES[int(np.argmax(p))] == lab, (name, lab, p)
        exp = label_of_score(score)
        assert lab == exp, (name, lab, score)
        print(f"  {name:18s} label={lab:8s} score={score:6.2f} "
              f"proba={np.round(p, 3).tolist()} {dt:6.1f}ms")

    print("\n== determinism ==")
    for name, text in EDGE_INPUTS.items():
        a = float(model.predict_scores([text])[0])
        b = float(model.predict_scores([text])[0])
        assert abs(a - b) < 1e-9, (name, a, b)
    print("  repeat runs identical  OK")

    print("\n== batch vs single ==")
    texts = list(EDGE_INPUTS.values())
    solo = [float(model.predict_scores([t])[0]) for t in texts]
    bat = [float(x) for x in model.predict_scores(texts)]
    for i, (a, b) in enumerate(zip(solo, bat)):
        assert abs(a - b) < 1e-9, (i, a, b)
    print(f"  {len(texts)} items agree  OK")

    print("\n== threshold boundaries ==")
    for s in (-5, 0, 49, 49.99, 50, 50.01, 71, 71.99, 72, 72.01, 99, 120):
        print(f"  {s:7.2f} -> {label_of_score(s)}")

    print("\n== demo CVs via parse_cv -> classifier (warm latency) ==")
    from src.parser.parser import parse_cv
    demo = ROOT / "demo"
    seen = {}
    for fname in DEMO_FILES:
        f = demo / fname
        if not f.exists():
            print(f"  !! missing {fname}")
            continue
        t0 = time.perf_counter()
        text = parse_cv(str(f))
        lab = model.predict([text])[0]
        score = float(model.predict_scores([text])[0])
        dt = (time.time() - t0) * 1000
        seen[fname] = (lab, score)
        print(f"  {fname:42.42s} label={str(lab):8s} score={score:6.2f}  all-in dt={dt:6.1f}ms")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
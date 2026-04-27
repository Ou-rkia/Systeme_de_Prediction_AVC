# =============================================================
# run_pipeline.py — SCRIPT MAÎTRE D'EXÉCUTION LOCALE
# Compatible : VS Code Local | CPU only
#
# Usage :
#   python run_pipeline.py          → pipeline complet
#   python run_pipeline.py --step 1 → étape 1 seulement
#   python run_pipeline.py --step 4 → étape 4 seulement
# =============================================================

import sys
import time
import argparse
from pathlib import Path


def banner(step, title):
    print(f"\n{'='*60}")
    print(f"  ÉTAPE {step} : {title}")
    print(f"{'='*60}")


def run_step1():
    banner(1, "GÉNÉRATION DES DONNÉES")
    import etape1_generation_donnees as e1
    return e1.generate_dataset()


def run_step2():
    banner(2, "EDA — ANALYSE EXPLORATOIRE")
    import etape2_eda as e2
    return e2.run_full_eda()


def run_step3():
    banner(3, "PRÉPROCESSING")
    import etape3_preprocessing as e3
    return e3.run_preprocessing()


def run_step4():
    banner(4, "MODÉLISATION (2-8 min sur CPU)")
    import etape4_modelisation as e4
    return e4.run_modeling()


def run_step5():
    banner(5, "TRACKING MLFLOW")
    import etape5_mlflow as e5
    return e5.run_mlflow_tracking()


def run_step6():
    banner(6, "GÉNÉRATION DES MODULES src/")
    exec(open("etape6_structure_projet.py").read())


def run_step7_test():
    banner(7, "TEST DE L'API (sans serveur)")
    import etape7_api_fastapi as api
    api.load_artifacts()

    from fastapi.testclient import TestClient
    client = TestClient(api.app)

    # Test health
    r = client.get("/health")
    print(f"  /health → {r.status_code}  {r.json()['status']}")

    # Test predict
    sample = {
        "age": 67.0, "gender": "Male", "hypertension": 1,
        "heart_disease": 1, "ever_married": "Yes",
        "work_type": "Private", "Residence_type": "Urban",
        "avg_glucose_level": 228.69, "bmi": 36.6,
        "smoking_status": "formerly smoked"
    }
    r = client.post("/predict", json=sample)
    pred = r.json()["prediction"]
    print(f"  /predict → {r.status_code}")
    print(f"    stroke_label : {pred['stroke_label']}")
    print(f"    stroke_risk  : {pred['stroke_risk']}")
    print(f"    confidence   : {pred['confidence']}")


def run_step8():
    banner(8, "TESTS AUTOMATISÉS")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "etape8_tests.py", "-v", "--tb=short"],
        capture_output=False
    )
    return result.returncode == 0


def run_step9():
    banner(9, "GÉNÉRATION FICHIERS DOCKER")
    exec(open("etape9_docker.py").read())


# ── Pipeline complet ──────────────────────────────────────────
def run_all():
    total_start = time.time()

    steps = [
        (1, "Génération des données", run_step1),
        (2, "EDA",                    run_step2),
        (3, "Préprocessing",          run_step3),
        (4, "Modélisation",           run_step4),
        (5, "MLflow Tracking",        run_step5),
        (6, "Modules src/",           run_step6),
        (7, "Test API",               run_step7_test),
        (8, "Tests pytest",           run_step8),
        (9, "Docker",                 run_step9),
    ]

    results = []
    for num, name, fn in steps:
        t0 = time.time()
        try:
            fn()
            elapsed = round(time.time() - t0, 1)
            results.append((num, name, "✅", elapsed))
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            results.append((num, name, "❌", elapsed))
            print(f"\n  ⚠️  Erreur étape {num} : {e}")

    total = round(time.time() - total_start, 1)

    # Récapitulatif
    print("\n" + "="*60)
    print("  RÉCAPITULATIF D'EXÉCUTION")
    print("="*60)
    for num, name, status, t in results:
        print(f"  {status}  Étape {num} — {name:<30} ({t}s)")
    print(f"\n  Temps total : {total}s")
    print("\n  Pour lancer l'API :")
    print("    uvicorn etape7_api_fastapi:app --reload --port 8000")
    print("    → http://localhost:8000/docs")
    print("\n  Pour MLflow UI :")
    print("    mlflow ui  →  http://localhost:5000")


# ── Argument parsing ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline MLOps AVC")
    parser.add_argument("--step", type=int, choices=range(1, 10),
                        help="Exécuter une seule étape (1-9)")
    args = parser.parse_args()

    step_map = {
        1: run_step1, 2: run_step2, 3: run_step3,
        4: run_step4, 5: run_step5, 6: run_step6,
        7: run_step7_test, 8: run_step8, 9: run_step9,
    }

    if args.step:
        step_map[args.step]()
    else:
        run_all()

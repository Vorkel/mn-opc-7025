#!/usr/bin/env python3
"""
Script simple pour exécuter les tests unitaires
"""
import subprocess
import sys
import os

def run_tests() -> None:
    """Exécute les tests unitaires"""
    print("🧪 Exécution des tests unitaires...")

    # Tests unitaires
    print("\n📋 Tests unitaires BusinessScorer:")
    result = subprocess.run([
        "poetry", "run", "pytest", "tests/unit/test_business_score.py", "-v"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Tests BusinessScorer: SUCCÈS")
    else:
        print("❌ Tests BusinessScorer: ÉCHEC")
        print(result.stdout)
        print(result.stderr)

    # Tests de validation des données
    print("\n📋 Tests de validation des données:")
    result = subprocess.run([
        "poetry", "run", "pytest", "tests/unit/test_data_validation.py", "-v", "--tb=short"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Tests validation données: SUCCÈS")
    else:
        print("❌ Tests validation données: ÉCHEC")
        print(result.stdout)
        print(result.stderr)

    # Tests feature engineering
    print("\n📋 Tests feature engineering:")
    result = subprocess.run([
        "poetry", "run", "pytest", "tests/unit/test_feature_engineering.py", "-v", "--tb=short"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Tests feature engineering: SUCCÈS")
    else:
        print("❌ Tests feature engineering: ÉCHEC")
        print(result.stdout)
        print(result.stderr)

    print("\n🎯 Résumé des tests terminé!")

if __name__ == "__main__":
    run_tests()

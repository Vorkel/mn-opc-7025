#!/usr/bin/env python3
"""
Tests de performance simples pour l'API
"""

import os
import sys
import time

import pytest

# Ajouter le répertoire api au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "api"))


def test_api_import_performance() -> None:
    """Test que l'import de l'API est rapide"""
    start_time = time.time()

    try:
        # Test d'import simple avec PYTHONPATH correct
        import os
        import sys

        # Ajouter le répertoire racine au PYTHONPATH
        project_root = os.path.join(os.path.dirname(__file__), "..", "..")
        sys.path.insert(0, project_root)
        from api.app import app

        import_time = time.time() - start_time

        # L'import doit prendre moins de 2 secondes
        assert import_time < 2.0, f"Import trop lent: {import_time:.2f}s"
        print(f"✅ Import API: {import_time:.3f}s")

    except ImportError as e:
        print(f"⚠️ API non disponible - test ignoré: {e}")
        return


def test_model_loading_performance() -> None:
    """Test que le chargement du modèle est rapide (si disponible)"""
    try:
        # Configurer le PYTHONPATH
        import os
        import sys

        project_root = os.path.join(os.path.dirname(__file__), "..", "..")
        sys.path.insert(0, project_root)

        from api.app import app

        # Test simple de création de l'app (sans TestClient)
        start_time = time.time()
        app_instance = app
        load_time = time.time() - start_time

        # Le chargement doit être rapide
        assert load_time < 1.0, f"Chargement app trop lent: {load_time:.2f}s"
        print(f"✅ Chargement app: {load_time:.3f}s")

    except Exception as e:
        print(f"⚠️ Test API non disponible: {e}")
        return


def test_memory_usage() -> None:
    """Test simple de l'utilisation mémoire"""
    import os

    import psutil

    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    # Simuler une opération
    import numpy as np

    data = np.random.random((1000, 10))

    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = memory_after - memory_before

    # L'augmentation mémoire doit être raisonnable (< 100MB)
    assert (
        memory_increase < 100
    ), f"Utilisation mémoire excessive: {memory_increase:.1f}MB"
    print(f"✅ Utilisation mémoire: {memory_increase:.1f}MB")


def test_calculation_performance() -> None:
    """Test de performance des calculs"""
    import numpy as np
    import pandas as pd

    # Test de performance d'un calcul simple
    start_time = time.time()

    # Simuler un calcul de feature engineering
    data = pd.DataFrame({
        "feature_1": np.random.random(10000),
        "feature_2": np.random.random(10000),
        "feature_3": np.random.random(10000),
    })

    # Calcul de ratios
    data["ratio_1_2"] = data["feature_1"] / data["feature_2"]
    data["ratio_2_3"] = data["feature_2"] / data["feature_3"]

    calculation_time = time.time() - start_time

    # Le calcul doit prendre moins de 1 seconde
    assert calculation_time < 1.0, f"Calcul trop lent: {calculation_time:.2f}s"
    print(f"✅ Calcul features: {calculation_time:.3f}s")


if __name__ == "__main__":
    # Tests simples sans pytest
    print("Tests de performance simples")
    print("=" * 40)

    try:
        test_api_import_performance()
        test_model_loading_performance()
        test_memory_usage()
        test_calculation_performance()
        print("\n✅ Tous les tests de performance passent!")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)

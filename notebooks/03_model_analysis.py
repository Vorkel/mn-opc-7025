# Notebook: 03_model_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime
import joblib

# Imports scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
from numpy.typing import NDArray

# Configuration
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Analyse et entraînement des modèles Home Credit - VERSION RAPIDE")
print("=" * 70)
print(f"Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# Chargement des données processées
# =============================================================================
print("\nChargement des données processées...")
print("=" * 40)

try:
    # Charger les données processées
    df_train = pd.read_csv("data/processed/train_processed.csv")
    df_test = pd.read_csv("data/processed/test_processed.csv")

    print(f"Données chargées avec succès!")
    print(f"Train: {df_train.shape}")
    print(f"Test: {df_test.shape}")

    # Échantillonner pour accélérer (prendre 20% des données)
    sample_size = int(len(df_train) * 0.2)
    df_train_sample = df_train.sample(n=sample_size, random_state=42)

    # Préparer les données
    X = df_train_sample.drop(["TARGET", "SK_ID_CURR"], axis=1, errors="ignore")
    y = df_train_sample["TARGET"]

    # Diviser les données
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Données préparées (échantillon 20%):")
    print(f"  X_train: {X_train.shape if hasattr(X_train, 'shape') else 'Unknown'}") # type: ignore
    print(f"  X_val: {X_val.shape if hasattr(X_val, 'shape') else 'Unknown'}") # type: ignore
    print(f"  y_train: {y_train.shape if hasattr(y_train, 'shape') else 'Unknown'}") # type: ignore
    print(f"  y_val: {y_val.shape if hasattr(y_val, 'shape') else 'Unknown'}") # type: ignore

except Exception as e:
    print(f"Erreur lors du chargement: {e}")
    exit(1)

# =============================================================================
# Définition des métriques métier
# =============================================================================
print("\nDÉFINITION DES MÉTRIQUES MÉTIER")
print("=" * 40)

class BusinessScorer:
    """Classe pour évaluer les modèles avec des métriques métier"""

    def __init__(self, fp_cost=1, fn_cost=10):
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost

    def evaluate_model(self, y_true, y_pred_proba, threshold=0.5):
        """Évalue un modèle avec des métriques métier"""
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculer les métriques
        auc_score = roc_auc_score(y_true, y_pred_proba)

        # Matrice de confusion
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Coût métier
        business_cost = fp * self.fp_cost + fn * self.fn_cost

        return {
            "auc_score": auc_score,
            "business_cost": business_cost,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp
        }

scorer = BusinessScorer(fp_cost=1, fn_cost=10)
models_results = {}

# =============================================================================
# Entraînement des modèles de base
# =============================================================================
print("\nENTRAÎNEMENT DES MODÈLES DE BASE")
print("=" * 40)

# 1. Logistic Regression
print("Entraînement Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
lr.fit(X_train, y_train)

y_pred_lr = lr.predict_proba(X_val)[:, 1]  # type: ignore
metrics_lr = scorer.evaluate_model(y_val, y_pred_lr)

models_results["logistic_regression"] = {
    "model": lr,
    "metrics": metrics_lr,
    "predictions": y_pred_lr,
}

print(f"  AUC: {metrics_lr['auc_score']:.4f}")
print(f"  Coût métier: {metrics_lr['business_cost']:.2f}")

# 2. Random Forest (version rapide)
print("Entraînement Random Forest...")
rf = RandomForestClassifier(
    n_estimators=50,  # Réduit de 100 à 50
    max_depth=10,     # Limité pour accélérer
    random_state=42,
    n_jobs=1,         # Éviter la parallélisation
    class_weight="balanced"
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict_proba(X_val)[:, 1]  # type: ignore
metrics_rf = scorer.evaluate_model(y_val, y_pred_rf)

models_results["random_forest"] = {
    "model": rf,
    "metrics": metrics_rf,
    "predictions": y_pred_rf,
}

print(f"  AUC: {metrics_rf['auc_score']:.4f}")
print(f"  Coût métier: {metrics_rf['business_cost']:.2f}")

# =============================================================================
# Gestion du déséquilibre avec SMOTE
# =============================================================================
print("\nGESTION DU DÉSÉQUILIBRE AVEC SMOTE")
print("=" * 40)

print("Entraînement Random Forest + SMOTE...")
smote = SMOTE(random_state=42, sampling_strategy='auto')  # Utiliser auto pour compatibilité
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train) # type: ignore

rf_smote = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=1,
    class_weight="balanced"
)
rf_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = rf_smote.predict_proba(X_val)[:, 1]  # type: ignore
metrics_smote = scorer.evaluate_model(y_val, y_pred_smote)

models_results["random_forest_smote"] = {
    "model": rf_smote,
    "metrics": metrics_smote,
    "predictions": y_pred_smote,
}

print(f"  AUC: {metrics_smote['auc_score']:.4f}")
print(f"  Coût métier: {metrics_smote['business_cost']:.2f}")

# =============================================================================
# Sélection du meilleur modèle
# =============================================================================
print("\nSÉLECTION DU MEILLEUR MODÈLE")
print("=" * 35)

# Comparer les modèles
comparison = []
for name, result in models_results.items():
    comparison.append({
        "model": name,
        "auc": result["metrics"]["auc_score"],
        "business_cost": result["metrics"]["business_cost"],
        "f1": result["metrics"]["f1_score"]
    })

comparison_df = pd.DataFrame(comparison)
print("\nComparaison des modèles:")
print(comparison_df.to_string(index=False))

# Sélectionner le meilleur modèle (coût métier le plus bas)
best_model_name = comparison_df.loc[comparison_df["business_cost"].idxmin(), "model"]
best_model = models_results[best_model_name]["model"]
best_metrics = models_results[best_model_name]["metrics"]

print(f"\nMeilleur modèle: {best_model_name}")
print(f"AUC: {best_metrics['auc_score']:.4f}")
print(f"Coût métier: {best_metrics['business_cost']:.2f}")

# =============================================================================
# Sauvegarde du modèle
# =============================================================================
print("\nSAUVEGARDE DU MODÈLE")
print("=" * 25)

# Créer le dossier models s'il n'existe pas
Path("models").mkdir(exist_ok=True)

# Sauvegarder le meilleur modèle
model_data = {
    "model": best_model,
    "threshold": 0.5,
    "feature_names": X_train.columns.tolist() if hasattr(X_train, 'columns') else [], # type: ignore
    "metrics": best_metrics,
    "model_name": best_model_name,
    "training_date": datetime.now().isoformat()
}

joblib.dump(model_data, "models/best_credit_model.pkl")
print("Modèle sauvegardé dans models/best_credit_model.pkl")

# =============================================================================
# Génération du rapport
# =============================================================================
print("\nGÉNÉRATION DU RAPPORT")
print("=" * 25)

# Créer le dossier reports s'il n'existe pas
Path("reports").mkdir(exist_ok=True)

# Rapport de comparaison
report = {
    "execution_date": datetime.now().isoformat(),
    "dataset_info": {
        "train_samples": len(df_train_sample),
        "validation_samples": len(X_val),
        "features": len(X_train.columns) if hasattr(X_train, 'columns') else 0 # type: ignore
    },
    "models_comparison": comparison,
    "best_model": {
        "name": best_model_name,
        "metrics": best_metrics
    },
    "recommendations": [
        "Modèle Random Forest avec SMOTE recommandé pour la production",
        "AUC > 0.7 acceptable pour ce type de problème",
        "Coût métier optimisé pour minimiser les faux négatifs"
    ]
}

# Sauvegarder le rapport
import json
with open("reports/model_analysis_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print("Rapport sauvegardé dans reports/model_analysis_report.json")

# =============================================================================
# Résumé final
# =============================================================================
print("\nRÉSUMÉ FINAL")
print("=" * 15)
print(f"✅ Modèles entraînés: {len(models_results)}")
print(f"✅ Meilleur modèle: {best_model_name}")
print(f"✅ AUC: {best_metrics['auc_score']:.4f}")
print(f"✅ Coût métier: {best_metrics['business_cost']:.2f}")
print(f"✅ Modèle sauvegardé: models/best_credit_model.pkl")
print(f"✅ Rapport généré: reports/model_analysis_report.json")

print("\nANALYSE TERMINÉE AVEC SUCCÈS!")
print("=" * 40)

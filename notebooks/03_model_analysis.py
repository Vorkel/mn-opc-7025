# Notebook: 03_model_analysis.py
# Analyse et entraînement des modèles pour Home Credit Default Risk

"""
OBJECTIFS DE CE NOTEBOOK :
- Entraîner différents modèles de machine learning
- Comparer les performances avec métriques métier
- Optimiser les hyperparamètres
- Sélectionner le meilleur modèle
- Sauvegarder le modèle final
"""

# =============================================================================
# Cell 1: Configuration et imports
# =============================================================================
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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Configuration
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration des graphiques
try:
    plt.style.use("seaborn-v0_8")
except OSError:
    plt.style.use("seaborn")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)

print("Analyse et entraînement des modèles Home Credit")
print("=" * 50)
print(f"Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# Cell 2: Chargement des données processées
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

    # Préparer les données
    X = df_train.drop(["TARGET", "SK_ID_CURR"], axis=1, errors="ignore")
    y = df_train["TARGET"]

    # Diviser les données
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Données préparées:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val: {y_val.shape}")

except Exception as e:
    print(f"Erreur lors du chargement: {e}")
    exit(1)

# =============================================================================
# Cell 3: Définition des métriques métier
# =============================================================================
print("\nDÉFINITION DES MÉTRIQUES MÉTIER")
print("=" * 40)


class BusinessScorer:
    """Classe pour évaluer les modèles avec des métriques métier"""

    def __init__(self, fp_cost=1, fn_cost=10):
        """
        Args:
            fp_cost: Coût d'un faux positif (refuser un bon client)
            fn_cost: Coût d'un faux négatif (accepter un mauvais client)
        """
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost

    def evaluate_model(self, y_true, y_pred_proba, threshold=0.5):
        """Évalue un modèle avec métriques métier"""

        # Prédictions binaires
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Matrice de confusion
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Métriques standard
        auc_score = roc_auc_score(y_true, y_pred_proba)

        # Métriques métier
        business_cost = (fp * self.fp_cost) + (fn * self.fn_cost)

        # Métriques additionnelles
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "auc_score": auc_score,
            "business_cost": business_cost,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "threshold": threshold,
        }

    def find_optimal_threshold(self, y_true, y_pred_proba):
        """Trouve le seuil optimal pour minimiser le coût métier"""

        thresholds = np.arange(0.1, 0.9, 0.05)
        costs = []

        for threshold in thresholds:
            metrics = self.evaluate_model(y_true, y_pred_proba, threshold)
            costs.append(metrics["business_cost"])

        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]

        return optimal_threshold, costs[optimal_idx]


# Créer l'évaluateur métier
scorer = BusinessScorer(fp_cost=1, fn_cost=10)
print("Évaluateur métier créé avec coûts:")
print(f"  Faux positif (refuser bon client): {scorer.fp_cost}")
print(f"  Faux négatif (accepter mauvais client): {scorer.fn_cost}")

# =============================================================================
# Cell 4: Entraînement des modèles de base
# =============================================================================
print("\nENTRAÎNEMENT DES MODÈLES DE BASE")
print("=" * 40)

models_results = {}

# 1. Modèle de baseline (Logistic Regression)
print("Entraînement Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict_proba(X_val)[:, 1]
metrics_lr = scorer.evaluate_model(y_val, y_pred_lr)

models_results["logistic_regression"] = {
    "model": lr,
    "metrics": metrics_lr,
    "predictions": y_pred_lr,
}

print(f"  AUC: {metrics_lr['auc_score']:.4f}")
print(f"  Coût métier: {metrics_lr['business_cost']:.2f}")

# 2. Random Forest standard
print("Entraînement Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict_proba(X_val)[:, 1]
metrics_rf = scorer.evaluate_model(y_val, y_pred_rf)

models_results["random_forest"] = {
    "model": rf,
    "metrics": metrics_rf,
    "predictions": y_pred_rf,
}

print(f"  AUC: {metrics_rf['auc_score']:.4f}")
print(f"  Coût métier: {metrics_rf['business_cost']:.2f}")

# =============================================================================
# Cell 5: Gestion du déséquilibre des classes
# =============================================================================
print("\nGESTION DU DÉSÉQUILIBRE DES CLASSES")
print("=" * 45)

# 1. Random Forest avec SMOTE
print("Entraînement Random Forest + SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = rf_smote.predict_proba(X_val)[:, 1]
metrics_smote = scorer.evaluate_model(y_val, y_pred_smote)

models_results["random_forest_smote"] = {
    "model": rf_smote,
    "metrics": metrics_smote,
    "predictions": y_pred_smote,
}

print(f"  AUC: {metrics_smote['auc_score']:.4f}")
print(f"  Coût métier: {metrics_smote['business_cost']:.2f}")

# 2. Random Forest avec Under-sampling
print("Entraînement Random Forest + Under-sampling...")
undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

rf_under = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_under.fit(X_train_under, y_train_under)

y_pred_under = rf_under.predict_proba(X_val)[:, 1]
metrics_under = scorer.evaluate_model(y_val, y_pred_under)

models_results["random_forest_under"] = {
    "model": rf_under,
    "metrics": metrics_under,
    "predictions": y_pred_under,
}

print(f"  AUC: {metrics_under['auc_score']:.4f}")
print(f"  Coût métier: {metrics_under['business_cost']:.2f}")

# =============================================================================
# Cell 6: Optimisation des hyperparamètres
# =============================================================================
print("\nOPTIMISATION DES HYPERPARAMÈTRES")
print("=" * 40)

# Optimisation Random Forest
print("Optimisation Random Forest...")
param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

rf_opt = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")


# Utiliser une fonction de scoring personnalisée
def business_scorer(estimator, X, y):
    y_pred = estimator.predict_proba(X)[:, 1]
    metrics = scorer.evaluate_model(y, y_pred)
    return -metrics["business_cost"]  # Négatif car GridSearchCV maximise


grid_search_rf = GridSearchCV(
    rf_opt, param_grid_rf, cv=3, scoring=business_scorer, n_jobs=-1, verbose=1
)

grid_search_rf.fit(X_train, y_train)

# Évaluer le modèle optimisé
y_pred_opt = grid_search_rf.predict_proba(X_val)[:, 1]
metrics_opt = scorer.evaluate_model(y_val, y_pred_opt)

models_results["random_forest_optimized"] = {
    "model": grid_search_rf.best_estimator_,
    "metrics": metrics_opt,
    "predictions": y_pred_opt,
    "best_params": grid_search_rf.best_params_,
}

print(f"Meilleurs paramètres: {grid_search_rf.best_params_}")
print(f"  AUC: {metrics_opt['auc_score']:.4f}")
print(f"  Coût métier: {metrics_opt['business_cost']:.2f}")

# =============================================================================
# Cell 7: Comparaison des modèles
# =============================================================================
print("\nCOMPARAISON DES MODÈLES")
print("=" * 30)

# Créer un tableau de comparaison
comparison_data = []
for name, result in models_results.items():
    metrics = result["metrics"]
    comparison_data.append(
        {
            "Modèle": name,
            "AUC": metrics["auc_score"],
            "Coût métier": metrics["business_cost"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1_score"],
        }
    )

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values("Coût métier")

print("Comparaison des modèles (triés par coût métier):")
print(comparison_df.to_string(index=False, float_format="%.4f"))

# Visualisation de la comparaison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Graphique AUC
axes[0].bar(comparison_df["Modèle"], comparison_df["AUC"])
axes[0].set_title("Comparaison AUC")
axes[0].set_ylabel("AUC Score")
axes[0].tick_params(axis="x", rotation=45)

# Graphique coût métier
axes[1].bar(comparison_df["Modèle"], comparison_df["Coût métier"])
axes[1].set_title("Comparaison Coût Métier")
axes[1].set_ylabel("Coût Métier")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("reports/model_comparison.png", dpi=300, bbox_inches="tight")
print("Graphique de comparaison sauvegardé")

# =============================================================================
# Cell 8: Sélection du meilleur modèle
# =============================================================================
print("\nSÉLECTION DU MEILLEUR MODÈLE")
print("=" * 35)

# Trouver le meilleur modèle
best_model_name = comparison_df.iloc[0]["Modèle"]
best_model_result = models_results[best_model_name]
best_model = best_model_result["model"]
best_model_metrics = best_model_result["metrics"]

print(f"MEILLEUR MODÈLE: {best_model_name}")
print(f"Coût métier: {best_model_metrics['business_cost']:.2f}")
print(f"AUC: {best_model_metrics['auc_score']:.4f}")

# Trouver le seuil optimal pour le meilleur modèle
optimal_threshold, optimal_cost = scorer.find_optimal_threshold(
    y_val, best_model_result["predictions"]
)

print(f"Seuil optimal: {optimal_threshold:.3f}")
print(f"Coût optimal: {optimal_cost:.2f}")

# Évaluer avec le seuil optimal
best_model_metrics_optimal = scorer.evaluate_model(
    y_val, best_model_result["predictions"], optimal_threshold
)

print(f"\nAvec seuil optimal:")
print(f"  Precision: {best_model_metrics_optimal['precision']:.4f}")
print(f"  Recall: {best_model_metrics_optimal['recall']:.4f}")
print(f"  F1-Score: {best_model_metrics_optimal['f1_score']:.4f}")

# =============================================================================
# Cell 9: Validation croisée
# =============================================================================
print("\nVALIDATION CROISÉE")
print("=" * 20)

# Validation croisée avec métrique métier
cv_scores = cross_val_score(
    best_model, X_train, y_train, cv=5, scoring=business_scorer, n_jobs=-1
)

print(f"Scores de validation croisée:")
print(f"  Moyenne: {-cv_scores.mean():.2f}")
print(f"  Écart-type: {cv_scores.std():.2f}")
print(f"  Min: {-cv_scores.max():.2f}")
print(f"  Max: {-cv_scores.min():.2f}")

# =============================================================================
# Cell 10: Sauvegarde du modèle final
# =============================================================================
print("\nSAUVEGARDE DU MODÈLE FINAL")
print("=" * 35)

# Créer le dossier models/ si nécessaire
models_path = Path("models")
models_path.mkdir(exist_ok=True)

# Sauvegarder le modèle
joblib.dump(best_model, "models/best_credit_model.pkl")

# Sauvegarder les métadonnées
model_metadata = {
    "model_name": best_model_name,
    "training_date": datetime.now().isoformat(),
    "features_count": X_train.shape[1],
    "training_samples": X_train.shape[0],
    "validation_samples": X_val.shape[0],
    "optimal_threshold": optimal_threshold,
    "metrics": best_model_metrics_optimal,
    "cv_scores": {
        "mean": float(-cv_scores.mean()),
        "std": float(cv_scores.std()),
        "min": float(-cv_scores.max()),
        "max": float(-cv_scores.min()),
    },
}

joblib.dump(model_metadata, "models/model_metadata.pkl")

print(f"Modèle sauvegardé: models/best_credit_model.pkl")
print(f"Métadonnées sauvegardées: models/model_metadata.pkl")

# =============================================================================
# Cell 11: Prédictions sur l'ensemble de test
# =============================================================================
print("\nPRÉDICTIONS SUR L'ENSEMBLE DE TEST")
print("=" * 40)

# Préparer les données de test
X_test = df_test.drop(["SK_ID_CURR"], axis=1, errors="ignore")

# S'assurer que les colonnes correspondent
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0

X_test = X_test[X_train.columns]

# Faire les prédictions
test_predictions = best_model.predict_proba(X_test)[:, 1]

# Créer le fichier de soumission
submission = pd.DataFrame(
    {"SK_ID_CURR": df_test["SK_ID_CURR"], "TARGET": test_predictions}
)

submission.to_csv("data/processed/submission.csv", index=False)

print(f"Prédictions sauvegardées: data/processed/submission.csv")
print(f"Nombre de prédictions: {len(submission)}")

# Statistiques des prédictions
print(f"\nStatistiques des prédictions:")
print(f"  Moyenne: {test_predictions.mean():.4f}")
print(f"  Médiane: {np.median(test_predictions):.4f}")
print(f"  Min: {test_predictions.min():.4f}")
print(f"  Max: {test_predictions.max():.4f}")

# =============================================================================
# Cell 12: Génération du rapport final
# =============================================================================
print("\nGÉNÉRATION DU RAPPORT FINAL")
print("=" * 35)

# Créer le rapport complet
analysis_summary = {
    "execution_date": datetime.now().isoformat(),
    "best_model": {
        "name": best_model_name,
        "type": type(best_model).__name__,
        "optimal_threshold": optimal_threshold,
    },
    "performance": {
        "auc_score": best_model_metrics_optimal["auc_score"],
        "business_cost": best_model_metrics_optimal["business_cost"],
        "precision": best_model_metrics_optimal["precision"],
        "recall": best_model_metrics_optimal["recall"],
        "f1_score": best_model_metrics_optimal["f1_score"],
    },
    "cross_validation": {
        "mean_cost": float(-cv_scores.mean()),
        "std_cost": float(cv_scores.std()),
        "min_cost": float(-cv_scores.max()),
        "max_cost": float(-cv_scores.min()),
    },
    "data_info": {
        "training_samples": X_train.shape[0],
        "validation_samples": X_val.shape[0],
        "test_samples": X_test.shape[0],
        "features_count": X_train.shape[1],
    },
    "model_comparison": comparison_df.to_dict("records"),
    "confusion_matrix": best_model_metrics_optimal["confusion_matrix"],
}

# Sauvegarder le rapport
import json

with open("reports/model_analysis_summary.json", "w") as f:
    json.dump(analysis_summary, f, indent=2, default=str)

print("Rapport sauvegardé: reports/model_analysis_summary.json")

# =============================================================================
# Cell 13: Résumé final
# =============================================================================
print("\nRÉSUMÉ DE L'ANALYSE DES MODÈLES")
print("=" * 50)

print(f"MEILLEUR MODÈLE: {best_model_name}")
print(f"Coût métier: {best_model_metrics['business_cost']:.2f}")
print(f"AUC: {best_model_metrics['auc_score']:.4f}")
print(f"Seuil optimal: {optimal_threshold:.4f}")
print(f"Validation croisée: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

print(f"\nINTERPRÉTATION MÉTIER:")
cm = best_model_metrics_optimal["confusion_matrix"]
print(f"Vrais négatifs: {cm['tn']:,} (bons clients correctement identifiés)")
print(f"Faux positifs: {cm['fp']:,} (bons clients refusés - manque à gagner)")
print(f"Faux négatifs: {cm['fn']:,} (mauvais clients acceptés - PERTES)")
print(f"Vrais positifs: {cm['tp']:,} (mauvais clients correctement identifiés)")

print(f"\nPROCHAINES ÉTAPES:")
print(f"   1. Analyse SHAP détaillée (notebook 04)")
print(f"   2. Déploiement via API FastAPI")
print(f"   3. Interface Streamlit pour les tests")
print(f"   4. Monitoring avec détection de drift")

print("\nANALYSE DES MODÈLES TERMINÉE!")
print("=" * 50)
print("Meilleur modèle sélectionné et sauvegardé")
print("Métriques tracées et rapportés")
print("Prédictions prêtes pour soumission")
print("Résumé sauvegardé dans reports/model_analysis_summary.json")

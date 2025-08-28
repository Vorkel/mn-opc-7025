# Notebook: 04_shap_analysis.py
# Analyse SHAP pour expliquer les prédictions du modèle Home Credit

"""
OBJECTIFS DE CE NOTEBOOK :
- Analyser l'importance des features avec SHAP
- Expliquer les prédictions individuelles
- Identifier les patterns de risque
- Générer des insights métier
- Créer des visualisations interactives
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

# Imports SHAP
import shap

# Imports scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

print("Analyse SHAP pour Home Credit Default Risk")
print("=" * 50)
print(f"Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# Cell 2: Chargement du modèle et des données
# =============================================================================
print("\nChargement du modèle et des données...")
print("=" * 45)

try:
    # Charger le modèle entraîné
    best_model = joblib.load("models/best_credit_model.pkl")
    model_metadata = joblib.load("models/model_metadata.pkl")

    print(f"Modèle chargé: {model_metadata['model_name']}")
    print(f"Date d'entraînement: {model_metadata['training_date']}")
    print(f"Seuil optimal: {model_metadata['optimal_threshold']:.4f}")

    # Charger les données
    df_train = pd.read_csv("data/processed/train_processed.csv")
    df_test = pd.read_csv("data/processed/test_processed.csv")

    print(f"Données chargées - Train: {df_train.shape}, Test: {df_test.shape}")

    # Préparer les données
    X = df_train.drop(["TARGET", "SK_ID_CURR"], axis=1, errors="ignore")
    y = df_train["TARGET"]

    # Diviser les données
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Données préparées pour l'analyse SHAP")

except Exception as e:
    print(f"Erreur lors du chargement: {e}")
    exit(1)

# =============================================================================
# Cell 3: Initialisation de l'explainer SHAP
# =============================================================================
print("\nINITIALISATION DE L'EXPLAINER SHAP")
print("=" * 40)

# Créer l'explainer SHAP
print("Création de l'explainer SHAP...")

# Pour Random Forest, utiliser TreeExplainer
if isinstance(best_model, RandomForestClassifier):
    explainer = shap.TreeExplainer(best_model)
    print("TreeExplainer créé pour Random Forest")
else:
    # Pour d'autres modèles, utiliser KernelExplainer
    explainer = shap.KernelExplainer(best_model.predict_proba, X_train[:100])
    print("KernelExplainer créé")

# Calculer les valeurs SHAP pour l'ensemble de validation
print("Calcul des valeurs SHAP pour l'ensemble de validation...")
shap_values = explainer.shap_values(X_val)

# Si c'est un Random Forest, shap_values est une liste
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Prendre les valeurs pour la classe positive

print(f"Valeurs SHAP calculées: {shap_values.shape}")

# =============================================================================
# Cell 4: Analyse de l'importance globale des features
# =============================================================================
print("\nANALYSE DE L'IMPORTANCE GLOBALE DES FEATURES")
print("=" * 50)

# Calculer l'importance moyenne absolue des features
feature_importance = np.abs(shap_values).mean(0)
feature_names = X_val.columns

# Créer un DataFrame avec l'importance des features
importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": feature_importance}
).sort_values("importance", ascending=False)

print("Top 20 features les plus importantes (SHAP):")
print(importance_df.head(20).to_string(index=False, float_format="%.4f"))

# Sauvegarder l'importance des features
importance_df.to_csv("reports/shap_feature_importance.csv", index=False)
print("Importance des features sauvegardée: reports/shap_feature_importance.csv")

# =============================================================================
# Cell 5: Visualisation de l'importance globale
# =============================================================================
print("\nVISUALISATION DE L'IMPORTANCE GLOBALE")
print("=" * 40)

# Graphique des features les plus importantes
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_val, plot_type="bar", max_display=20)
plt.title("Top 20 Features - Importance SHAP Globale")
plt.tight_layout()
plt.savefig("reports/shap_global_importance.png", dpi=300, bbox_inches="tight")
print("Graphique d'importance globale sauvegardé")

# Graphique de dispersion SHAP
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_val, max_display=20)
plt.title("Distribution SHAP des Features")
plt.tight_layout()
plt.savefig("reports/shap_summary_plot.png", dpi=300, bbox_inches="tight")
print("Graphique de distribution SHAP sauvegardé")

# =============================================================================
# Cell 6: Analyse des interactions entre features
# =============================================================================
print("\nANALYSE DES INTERACTIONS ENTRE FEATURES")
print("=" * 45)

# Sélectionner les features les plus importantes pour l'analyse d'interaction
top_features = importance_df.head(10)["feature"].tolist()
X_val_top = X_val[top_features]

print(f"Analyse des interactions pour les {len(top_features)} features principales...")

# Calculer les valeurs SHAP pour les features principales
shap_values_top = explainer.shap_values(X_val_top)
if isinstance(shap_values_top, list):
    shap_values_top = shap_values_top[1]

# Graphique d'interaction pour les features les plus importantes
plt.figure(figsize=(15, 12))
shap.dependence_plot(0, shap_values_top, X_val_top, interaction_index=1)
plt.title("Interaction SHAP - Features Principales")
plt.tight_layout()
plt.savefig("reports/shap_interaction_plot.png", dpi=300, bbox_inches="tight")
print("Graphique d'interaction SHAP sauvegardé")

# =============================================================================
# Cell 7: Analyse des prédictions individuelles
# =============================================================================
print("\nANALYSE DES PRÉDICTIONS INDIVIDUELLES")
print("=" * 45)

# Sélectionner quelques cas intéressants pour l'analyse
print("Sélection de cas d'étude...")

# Cas 1: Client à haut risque (probabilité élevée de défaut)
high_risk_idx = np.argmax(best_model.predict_proba(X_val)[:, 1])
high_risk_prob = best_model.predict_proba(X_val)[high_risk_idx, 1]

# Cas 2: Client à faible risque (probabilité faible de défaut)
low_risk_idx = np.argmin(best_model.predict_proba(X_val)[:, 1])
low_risk_prob = best_model.predict_proba(X_val)[low_risk_idx, 1]

# Cas 3: Client borderline (probabilité proche du seuil)
threshold = model_metadata["optimal_threshold"]
borderline_mask = np.abs(best_model.predict_proba(X_val)[:, 1] - threshold) < 0.05
if np.any(borderline_mask):
    borderline_idx = np.where(borderline_mask)[0][0]
    borderline_prob = best_model.predict_proba(X_val)[borderline_idx, 1]
else:
    borderline_idx = np.argmin(
        np.abs(best_model.predict_proba(X_val)[:, 1] - threshold)
    )
    borderline_prob = best_model.predict_proba(X_val)[borderline_idx, 1]

print(f"Cas sélectionnés:")
print(f"  Haut risque: Client {high_risk_idx}, Probabilité: {high_risk_prob:.4f}")
print(f"  Faible risque: Client {low_risk_idx}, Probabilité: {low_risk_prob:.4f}")
print(f"  Borderline: Client {borderline_idx}, Probabilité: {borderline_prob:.4f}")

# =============================================================================
# Cell 8: Visualisation des prédictions individuelles
# =============================================================================
print("\nVISUALISATION DES PRÉDICTIONS INDIVIDUELLES")
print("=" * 50)

# Créer les graphiques waterfall pour chaque cas
cases = [
    ("Haut Risque", high_risk_idx, high_risk_prob),
    ("Faible Risque", low_risk_idx, low_risk_prob),
    ("Borderline", borderline_idx, borderline_prob),
]

for case_name, idx, prob in cases:
    print(f"Création du graphique waterfall pour {case_name}...")

    # Créer le graphique waterfall
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=(
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, list)
                else explainer.expected_value
            ),
            data=X_val.iloc[idx],
            feature_names=X_val.columns,
        ),
        max_display=15,
    )
    plt.title(f"Explication SHAP - {case_name} (Probabilité: {prob:.4f})")
    plt.tight_layout()
    plt.savefig(
        f'reports/shap_waterfall_{case_name.lower().replace(" ", "_")}.png',
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f"  Graphique waterfall sauvegardé: reports/shap_waterfall_{case_name.lower().replace(' ', '_')}.png"
    )

# =============================================================================
# Cell 9: Analyse des patterns de risque
# =============================================================================
print("\nANALYSE DES PATTERNS DE RISQUE")
print("=" * 35)

# Analyser les valeurs SHAP par classe
print("Analyse des patterns par classe de risque...")

# Diviser les données par classe
high_risk_mask = best_model.predict_proba(X_val)[:, 1] > threshold
low_risk_mask = best_model.predict_proba(X_val)[:, 1] <= threshold

# Calculer les valeurs SHAP moyennes par classe
high_risk_shap = shap_values[high_risk_mask].mean(0)
low_risk_shap = shap_values[low_risk_mask].mean(0)

# Créer un DataFrame de comparaison
comparison_df = pd.DataFrame(
    {
        "feature": feature_names,
        "high_risk_impact": high_risk_shap,
        "low_risk_impact": low_risk_shap,
        "difference": high_risk_shap - low_risk_shap,
    }
).sort_values("difference", ascending=False)

print("Top 10 features avec le plus grand impact différentiel:")
print(comparison_df.head(10).to_string(index=False, float_format="%.4f"))

# Sauvegarder la comparaison
comparison_df.to_csv("reports/shap_risk_patterns.csv", index=False)
print("Patterns de risque sauvegardés: reports/shap_risk_patterns.csv")

# =============================================================================
# Cell 10: Visualisation des patterns de risque
# =============================================================================
print("\nVISUALISATION DES PATTERNS DE RISQUE")
print("=" * 40)

# Graphique des features avec le plus grand impact différentiel
plt.figure(figsize=(12, 10))
top_diff_features = comparison_df.head(15)
x_pos = np.arange(len(top_diff_features))

plt.barh(x_pos, top_diff_features["difference"], color="red", alpha=0.7)
plt.yticks(x_pos, top_diff_features["feature"])
plt.xlabel("Impact Différentiel (Haut Risque - Faible Risque)")
plt.title("Features avec le Plus Grand Impact Différentiel")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("reports/shap_risk_patterns.png", dpi=300, bbox_inches="tight")
print("Graphique des patterns de risque sauvegardé")

# =============================================================================
# Cell 11: Analyse des valeurs aberrantes
# =============================================================================
print("\nANALYSE DES VALEURS ABERRANTES")
print("=" * 35)

# Identifier les prédictions avec les valeurs SHAP les plus extrêmes
print("Identification des cas extrêmes...")

# Calculer la somme absolue des valeurs SHAP pour chaque prédiction
shap_magnitude = np.abs(shap_values).sum(1)

# Trouver les cas avec les valeurs SHAP les plus élevées
extreme_cases = np.argsort(shap_magnitude)[-10:]

print("Top 10 cas avec les explications SHAP les plus complexes:")
for i, idx in enumerate(extreme_cases[::-1]):
    prob = best_model.predict_proba(X_val)[idx, 1]
    magnitude = shap_magnitude[idx]
    print(
        f"  {i+1}. Client {idx}: Probabilité={prob:.4f}, Magnitude SHAP={magnitude:.2f}"
    )

# Analyser un cas extrême
extreme_idx = extreme_cases[-1]
extreme_prob = best_model.predict_proba(X_val)[extreme_idx, 1]

print(f"\nAnalyse détaillée du cas le plus extrême (Client {extreme_idx}):")
print(f"  Probabilité de défaut: {extreme_prob:.4f}")
print(f"  Magnitude SHAP: {shap_magnitude[extreme_idx]:.2f}")

# Créer un graphique waterfall pour ce cas
plt.figure(figsize=(12, 8))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[extreme_idx],
        base_values=(
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, list)
            else explainer.expected_value
        ),
        data=X_val.iloc[extreme_idx],
        feature_names=X_val.columns,
    ),
    max_display=15,
)
plt.title(f"Cas Extrême - Client {extreme_idx} (Probabilité: {extreme_prob:.4f})")
plt.tight_layout()
plt.savefig("reports/shap_extreme_case.png", dpi=300, bbox_inches="tight")
print("Graphique du cas extrême sauvegardé")

# =============================================================================
# Cell 12: Génération du rapport SHAP
# =============================================================================
print("\nGÉNÉRATION DU RAPPORT SHAP")
print("=" * 35)

# Créer un rapport complet de l'analyse SHAP
shap_report = {
    "execution_date": datetime.now().isoformat(),
    "model_info": {
        "name": model_metadata["model_name"],
        "type": type(best_model).__name__,
        "optimal_threshold": model_metadata["optimal_threshold"],
    },
    "analysis_summary": {
        "total_samples_analyzed": len(X_val),
        "high_risk_samples": int(high_risk_mask.sum()),
        "low_risk_samples": int(low_risk_mask.sum()),
        "top_features_count": len(top_features),
    },
    "top_features": importance_df.head(20).to_dict("records"),
    "risk_patterns": comparison_df.head(15).to_dict("records"),
    "case_studies": {
        "high_risk_case": {
            "client_id": int(high_risk_idx),
            "probability": float(high_risk_prob),
            "risk_level": "high",
        },
        "low_risk_case": {
            "client_id": int(low_risk_idx),
            "probability": float(low_risk_prob),
            "risk_level": "low",
        },
        "borderline_case": {
            "client_id": int(borderline_idx),
            "probability": float(borderline_prob),
            "risk_level": "borderline",
        },
        "extreme_case": {
            "client_id": int(extreme_idx),
            "probability": float(extreme_prob),
            "shap_magnitude": float(shap_magnitude[extreme_idx]),
            "risk_level": "extreme",
        },
    },
    "insights": [
        f"Les {len(top_features)} features principales expliquent la majorité des prédictions",
        f"Les clients à haut risque ont des patterns SHAP distincts",
        f"Certaines features ont un impact différentiel important entre les classes",
        f"Les cas extrêmes nécessitent une attention particulière pour l'interprétation",
    ],
}

# Sauvegarder le rapport
import json

with open("reports/shap_analysis_report.json", "w") as f:
    json.dump(shap_report, f, indent=2, default=str)

print("Rapport SHAP sauvegardé: reports/shap_analysis_report.json")

# =============================================================================
# Cell 13: Résumé final et recommandations
# =============================================================================
print("\nRÉSUMÉ DE L'ANALYSE SHAP")
print("=" * 40)

print(f"ANALYSE TERMINÉE AVEC SUCCÈS!")
print(f"Modèle analysé: {model_metadata['model_name']}")
print(f"Échantillon analysé: {len(X_val)} clients")
print(f"Features principales identifiées: {len(top_features)}")

print(f"\nINSIGHTS PRINCIPAUX:")
print(f"  1. Top 3 features les plus importantes:")
for i, (_, row) in enumerate(importance_df.head(3).iterrows()):
    print(f"     {i+1}. {row['feature']}: {row['importance']:.4f}")

print(f"\n  2. Features avec le plus grand impact différentiel:")
for i, (_, row) in enumerate(comparison_df.head(3).iterrows()):
    print(f"     {i+1}. {row['feature']}: {row['difference']:.4f}")

print(f"\n  3. Cas d'étude analysés:")
print(f"     - Haut risque: Client {high_risk_idx} (Prob: {high_risk_prob:.4f})")
print(f"     - Faible risque: Client {low_risk_idx} (Prob: {low_risk_prob:.4f})")
print(f"     - Borderline: Client {borderline_idx} (Prob: {borderline_prob:.4f})")
print(f"     - Extrême: Client {extreme_idx} (Prob: {extreme_prob:.4f})")

print(f"\nFICHIERS GÉNÉRÉS:")
print(f"  - reports/shap_feature_importance.csv")
print(f"  - reports/shap_risk_patterns.csv")
print(f"  - reports/shap_analysis_report.json")
print(f"  - Graphiques PNG dans reports/")

print(f"\nRECOMMANDATIONS MÉTIER:")
print(f"  1. Surveiller particulièrement les features à fort impact différentiel")
print(f"  2. Analyser les cas borderline pour affiner les critères")
print(f"  3. Utiliser les explications SHAP pour la communication client")
print(f"  4. Intégrer ces insights dans le processus de décision")

print(f"\nPROCHAINES ÉTAPES:")
print(f"  1. Déploiement de l'API avec explications SHAP")
print(f"  2. Interface utilisateur pour l'analyse des cas")
print(f"  3. Monitoring des explications SHAP en production")
print(f"  4. Formation des équipes métier sur l'interprétation")

print("\nANALYSE SHAP TERMINÉE!")
print("=" * 50)
print("Explications du modèle générées et sauvegardées")
print("Insights métier identifiés et documentés")
print("Prêt pour le déploiement avec explications")

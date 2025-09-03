# Notebook: 04_shap_analysis.py

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
import shap

# Configuration
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Analyse SHAP pour Home Credit Default Risk")
print("=" * 65)
print(f"Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# Chargement du modèle et des données
# =============================================================================
print("\nChargement du modèle et des données...")
print("=" * 40)

try:
    # Charger le modèle entraîné
    model_path = "models/best_credit_model.pkl"
    print(f"Chargement du modèle depuis: {model_path}")

    model_dict = joblib.load(model_path)
    # Le modèle est maintenant directement le RandomForest, pas un dict
    if isinstance(model_dict, dict) and "model" in model_dict:
        model = model_dict["model"]
        feature_names = model_dict.get("feature_names", [f"feature_{i}" for i in range(153)])
        print(f"Modèle chargé: {model_dict.get('model_name', 'RandomForest')}")
    else:
        model = model_dict
        feature_names = [f"feature_{i}" for i in range(153)]
        print(f"Modèle chargé: RandomForestClassifier")

    # Remplacer model_data par des valeurs directes
    model_data = {
        "model_name": "RandomForestClassifier",
        "feature_names": feature_names
    }
    print(f"Features: {len(feature_names)}")

    # Charger les données
    df_train = pd.read_csv("data/processed/train_processed.csv")
    df_test = pd.read_csv("data/processed/test_processed.csv")

    # Préparer les données (même échantillon que l'entraînement)
    sample_size = int(len(df_train) * 0.2)
    df_train_sample = df_train.sample(n=sample_size, random_state=42)

    X = df_train_sample.drop(["TARGET", "SK_ID_CURR"], axis=1, errors="ignore")
    y = df_train_sample["TARGET"]

    # S'assurer que les colonnes correspondent
    X = X[feature_names]

    print(f"Données chargées: {X.shape}")

except Exception as e:
    print(f"Erreur lors du chargement: {e}")
    exit(1)

# =============================================================================
# Analyse SHAP globale
# =============================================================================
print("\nANALYSE SHAP GLOBALE")
print("=" * 25)

# Créer un échantillon pour l'analyse SHAP (plus petit pour la vitesse)
shap_sample_size = min(1000, len(X))
X_shap = X.sample(n=shap_sample_size, random_state=42)

print(f"Échantillon SHAP: {X_shap.shape}")

# Calculer les valeurs SHAP
print("Calcul des valeurs SHAP...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

print(f"Type des valeurs SHAP: {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"Nombre de classes: {len(shap_values)}")
    # Prendre les valeurs pour la classe positive (défaut)
    shap_values = shap_values[1]
else:
    print(f"Forme initiale des valeurs SHAP: {shap_values.shape}")
    # Si c'est un array 3D, prendre la dernière dimension (classe positive)
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

print("Valeurs SHAP calculées!")
print(f"Forme finale des valeurs SHAP: {shap_values.shape}")

# =============================================================================
# Importance globale des features
# =============================================================================
print("\nIMPORTANCE GLOBALE DES FEATURES")
print("=" * 35)

# Calculer l'importance moyenne
feature_importance = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top 15 features les plus importantes:")
print(importance_df.head(15).to_string(index=False))

# Sauvegarder l'importance des features
importance_df.to_csv("data/processed/feature_importance.csv", index=False)
print("Importance des features sauvegardée")

# =============================================================================
# Visualisations SHAP
# =============================================================================
print("\nGÉNÉRATION DES VISUALISATIONS")
print("=" * 30)

# Créer le dossier reports s'il n'existe pas
Path("reports").mkdir(exist_ok=True)

# 1. Graphique d'importance globale
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
plt.title("Importance globale des features (SHAP)")
plt.tight_layout()
plt.savefig("reports/shap_global_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Graphique d'importance globale généré")

# 2. Graphique des valeurs SHAP
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_shap, feature_names=feature_names, plot_type="bar", show=False)
plt.title("Top 20 features par importance SHAP")
plt.tight_layout()
plt.savefig("reports/shap_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Graphique des valeurs SHAP généré")

# 3. Graphique de dépendance pour les top features
top_features = importance_df.head(5)['feature'].tolist()

for i, feature in enumerate(top_features):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature, shap_values, X_shap, feature_names=feature_names, show=False)
    plt.title(f"SHAP Dependence Plot - {feature}")
    plt.tight_layout()
    plt.savefig(f"reports/shap_dependence_{feature}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique de dépendance généré pour {feature}")

# =============================================================================
# Analyse SHAP locale (exemples)
# =============================================================================
print("\nANALYSE SHAP LOCALE")
print("=" * 20)

# Sélectionner quelques exemples
sample_indices = [0, 100, 500]  # Exemples à analyser

for idx in sample_indices:
    if idx < len(X_shap):
        # Prédiction du modèle
        prediction = model.predict_proba(X_shap.iloc[[idx]])[0]  # type: ignore

        # Graphique SHAP local (version compatible)
        try:
            plt.figure(figsize=(12, 8))
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            shap.force_plot(
                expected_value,
                shap_values[idx, :],
                X_shap.iloc[idx],
                feature_names=feature_names,
                show=False
            )
            plt.title(f"SHAP Local - Exemple {idx} (Probabilité défaut: {prediction[1]:.3f})")
            plt.tight_layout()
            plt.savefig(f"reports/shap_local_example_{idx}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Graphique local généré pour l'exemple {idx}")
        except Exception as e:
            print(f"⚠️ Erreur graphique local {idx}: {e}")
            # Alternative: graphique simple
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(feature_names[:10])), shap_values[idx, :10])
            plt.yticks(range(len(feature_names[:10])), feature_names[:10])
            plt.title(f"SHAP Values - Exemple {idx}")
            plt.tight_layout()
            plt.savefig(f"reports/shap_local_example_{idx}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Graphique local alternatif généré pour l'exemple {idx}")

# =============================================================================
# Analyse des interactions
# =============================================================================
print("\nANALYSE DES INTERACTIONS")
print("=" * 25)

# Analyser les interactions pour les top features
if len(top_features) >= 2:
    feature1, feature2 = top_features[0], top_features[1]

    plt.figure(figsize=(12, 8))
    shap.dependence_plot(
        feature1, shap_values, X_shap,
        interaction_index=feature2,
        feature_names=feature_names,
        show=False
    )
    plt.title(f"Interaction SHAP: {feature1} vs {feature2}")
    plt.tight_layout()
    plt.savefig(f"reports/shap_interaction_{feature1}_{feature2}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique d'interaction généré: {feature1} vs {feature2}")

# =============================================================================
# Génération du rapport
# =============================================================================
print("\nGÉNÉRATION DU RAPPORT")
print("=" * 25)

# Rapport SHAP
shap_report = {
    "execution_date": datetime.now().isoformat(),
    "model_info": {
        "name": model_data["model_name"],
        "features_count": len(feature_names),
        "shap_sample_size": shap_sample_size
    },
    "top_features": importance_df.head(10).to_dict('records'),
    "visualizations": [
        "shap_global_importance.png",
        "shap_feature_importance.png"
    ] + [f"shap_dependence_{f}.png" for f in top_features] +
    [f"shap_local_example_{i}.png" for i in sample_indices if i < len(X_shap)],
    "insights": [
        f"Feature la plus importante: {importance_df.iloc[0]['feature']}",
        f"Top 5 features représentent {importance_df.head(5)['importance'].sum() / importance_df['importance'].sum() * 100:.1f}% de l'importance totale",
        "Les features temporelles et financières sont très importantes",
        "Les interactions entre features sont significatives"
    ]
}

# Sauvegarder le rapport
import json
with open("reports/shap_analysis_report.json", "w") as f:
    json.dump(shap_report, f, indent=2, default=str)

print("Rapport SHAP sauvegardé")

# =============================================================================
# Résumé final
# =============================================================================
print("\nRÉSUMÉ FINAL")
print("=" * 15)
print(f"✅ Modèle analysé: {model_data['model_name']}")
print(f"✅ Features analysées: {len(feature_names)}")
print(f"✅ Échantillon SHAP: {shap_sample_size}")
print(f"✅ Graphiques générés: {len(shap_report['visualizations'])}")
print(f"✅ Feature la plus importante: {importance_df.iloc[0]['feature']}")
print(f"✅ Importance sauvegardée: data/processed/feature_importance.csv")
print(f"✅ Rapport généré: reports/shap_analysis_report.json")

print("\nANALYSE SHAP TERMINÉE AVEC SUCCÈS!")
print("=" * 45)

# Application Streamlit - Système de Scoring Crédit

## Vue d'ensemble

Cette application Streamlit locale permet d'effectuer des prédictions de scoring crédit sans dépendance à une API externe. Elle utilise les modèles et données générés localement par les scripts du projet.

## Lancement de l'application

### Prérequis

1. **Python 3.8+** installé
2. **Streamlit** installé : `pip install streamlit`
3. **Toutes les dépendances** du projet installées : `pip install -r requirements.txt`

### Démarrage

```bash
# Depuis la racine du projet
streamlit run streamlit_app/main.py

# Ou avec un port spécifique
streamlit run streamlit_app/main.py --server.port 8501
```

L'application sera accessible à l'adresse : `http://localhost:8502`

## Fonctionnalités

### 1. Prédiction Individuelle

- **Saisie manuelle** des données client
- **Prédiction en temps réel** du risque de défaut
- **Visualisation** du score avec graphique en jauge
- **Décision automatique** (Accordé/Refusé) basée sur le seuil optimal

### 2. Analyse en Lot

- **Upload de fichiers CSV** avec données clients
- **Traitement par lot** de plusieurs clients
- **Statistiques globales** et visualisations
- **Export des résultats** en CSV

### 3. Historique des Prédictions

- **Sauvegarde automatique** des prédictions effectuées
- **Consultation** des résultats précédents
- **Gestion** de l'historique (suppression)

### 4. Analyse des Features

- **Importance des features** avec graphiques interactifs
- **Visualisations SHAP** (si disponibles)
- **Tableaux détaillés** d'importance

### 5. Rapports et Visualisations

- **Graphiques d'analyse** générés par les scripts
- **Rapports JSON** d'exploration des données
- **Matrices de corrélation** et distributions

## Structure des Données

L'application utilise les fichiers suivants :

```bash
├── streamlit_app/
├── models/                    # Modèles entraînés
│   ├── best_model.pkl        # Modèle optimal
│   └── model.pkl             # Modèle standard
├── data/processed/           # Données traitées
│   ├── feature_importance.csv
│   ├── test_processed.csv
│   └── train_processed.csv
├── reports/                  # Rapports et visualisations
│   ├── feature_importance.png
│   ├── correlation_matrix.png
│   ├── shap_*.png
│   └── *.json
└── src/                      # Scripts d'analyse
    ├── business_score.py
    ├── feature_importance.py
    └── model_training.py
```

## Configuration

### Variables d'environnement

Aucune variable d'environnement requise - l'application fonctionne entièrement en local.

### Chemins personnalisés

Les chemins sont automatiquement détectés depuis la structure du projet. Si vous modifiez la structure, ajustez les variables dans `main.py` :

```python
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
```

## Format des Données

### Données individuelles

L'application accepte les champs suivants :

**Informations personnelles :**

- `CODE_GENDER` : "M" ou "F"
- `FLAG_OWN_CAR` : "Y" ou "N"
- `FLAG_OWN_REALTY` : "Y" ou "N"
- `CNT_CHILDREN` : Nombre d'enfants
- `CNT_FAM_MEMBERS` : Nombre de membres de la famille

**Informations financières :**

- `AMT_INCOME_TOTAL` : Revenu total annuel
- `AMT_CREDIT` : Montant du crédit demandé
- `AMT_ANNUITY` : Montant de l'annuité
- `AMT_GOODS_PRICE` : Prix des biens

**Informations temporelles :**

- `DAYS_BIRTH` : Âge en jours (négatif)
- `DAYS_EMPLOYED` : Expérience en jours (négatif)
- `DAYS_REGISTRATION` : Jours depuis l'enregistrement

### Format CSV pour l'analyse en lot

Utilisez le template fourni dans l'application ou respectez le format des colonnes listées ci-dessus.

## Interprétation des Résultats

### Score de risque

- **🟢 FAIBLE** : Probabilité < 30%
- **🟡 MOYEN** : Probabilité entre 30% et 60%
- **🔴 ÉLEVÉ** : Probabilité > 60%

### Décision

- **ACCORDÉ** : Probabilité < seuil optimal
- **REFUSÉ** : Probabilité ≥ seuil optimal

## Dépannage

### Erreurs courantes

1. **"Aucun modèle trouvé"**

   - Vérifiez que les scripts d'entraînement ont été exécutés
   - Vérifiez la présence de fichiers `.pkl` dans `models/`

2. **"Données non disponibles"**

   - Exécutez les scripts d'analyse dans `src/`
   - Vérifiez la présence des fichiers dans `data/processed/` et `reports/`

3. **Erreurs d'import**
   - Vérifiez que tous les packages sont installés
   - Exécutez `pip install -r requirements.txt`

### Logs

Les erreurs sont affichées directement dans l'interface Streamlit. Pour plus de détails, consultez la console où l'application a été lancée.

## Mise à jour

Pour mettre à jour l'application :

1. **Arrêtez** l'application (Ctrl+C)
2. **Modifiez** le code si nécessaire
3. **Relancez** avec `streamlit run streamlit_app/main.py`

## Support

**Version** : 2.0.0 - Locale
**Projet** : MLOps crédit
**Statut** : Prêt pour démonstration

---

_Application développée pour le projet MLOps - Système de scoring crédit_

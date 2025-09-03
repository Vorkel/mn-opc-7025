# MLflow - Guide d'utilisation

## Vue d'ensemble

MLflow est configuré dans ce projet pour le tracking des expérimentations de machine learning. Il permet de :

- **Tracker les métriques** : Business Score, AUC, Accuracy, etc.
- **Comparer les modèles** : Visualisation des performances
- **Gérer les versions** : Registry des modèles
- **Documenter les expérimentations** : Paramètres, métriques, artifacts

## Accès rapide

### Interface web

```bash
# Lancer l'interface
./scripts/launch_mlflow.sh

# Ou manuellement
mlflow ui --host 0.0.0.0 --port 5000
```

**URL d'accès** : http://localhost:5000 (ou port suivant si 5000 occupé)

### Vérification de l'état

```bash
# Vérifier l'installation et les runs
python scripts/check_mlflow_status.py
```

## État actuel du projet

### Installation

- ✅ **MLflow version** : 3.1.4
- ✅ **Dossier mlruns** : Présent avec données
- ✅ **Expérimentations** : 1 expérimentation active
- ✅ **Runs** : 5 runs disponibles

### Métriques trackées

- `test_metric` : Métrique de test
- `accuracy` : Précision du modèle
- `true_positives` : Vrais positifs
- `f1_score` : Score F1
- `false_positives` : Faux positifs

### Paramètres trackés

- `test_param` : Paramètre de test
- `n_estimators` : Nombre d'estimateurs
- `sampling_strategy` : Stratégie d'échantillonnage
- `model_type` : Type de modèle
- `baseline` : Modèle de référence

## Configuration

### Structure des dossiers

```
mlruns/
├── 186361857841028067/          # Expérimentation principale
│   ├── meta.yaml               # Métadonnées
│   └── [run_id]/               # Runs individuels
│       ├── meta.yaml           # Métadonnées du run
│       ├── metrics/            # Métriques
│       ├── params/             # Paramètres
│       └── artifacts/          # Fichiers générés
```

### Scripts disponibles

- `scripts/launch_mlflow.sh` : Lancement automatique de l'interface
- `scripts/check_mlflow_status.py` : Vérification de l'état MLflow

## Utilisation dans le code

### Tracking des métriques

```python
import mlflow

# Démarrer un run
with mlflow.start_run():
    # Logger des métriques
    mlflow.log_metric("business_score", 0.85)
    mlflow.log_metric("auc_score", 0.736)

    # Logger des paramètres
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)

    # Logger des artifacts
    mlflow.log_artifact("reports/feature_importance.png")
```

### Comparaison des runs

```python
# Comparer les runs
runs = mlflow.search_runs()
best_run = runs.loc[runs['metrics.business_score'].idxmax()]
```

## Interface web

### Fonctionnalités principales

1. **Dashboard** : Vue d'ensemble des expérimentations
2. **Runs** : Liste détaillée des runs
3. **Comparaison** : Comparaison de plusieurs runs
4. **Artifacts** : Visualisation des fichiers générés

### Navigation

- **Expérimentations** : Sélectionner l'expérimentation à analyser
- **Runs** : Cliquer sur un run pour voir les détails
- **Métriques** : Graphiques des métriques au fil du temps
- **Paramètres** : Tableau des paramètres utilisés

## Dépannage

### Port déjà utilisé

```bash
# Vérifier les processus
lsof -i :5000

# Utiliser un autre port
./scripts/launch_mlflow.sh 8080
```

### Dossier mlruns manquant

```bash
# Créer le dossier
mkdir -p mlruns

# Vérifier les permissions
ls -la mlruns/
```

### Erreur de connexion

```bash
# Vérifier l'installation
pip list | grep mlflow

# Réinstaller si nécessaire
pip install mlflow
```

## Documentation

### Guides disponibles

- `docs/mlflow_ui_guide.md` : Guide détaillé de l'interface
- `reports/mlflow_status_report.json` : Rapport d'état automatique

### Ressources externes

- [Documentation officielle MLflow](https://mlflow.org/docs/latest/index.html)
- [Guide du tracking](https://mlflow.org/docs/latest/tracking.html)
- [API Python](https://mlflow.org/docs/latest/python_api/index.html)

## Sécurité

### Accès réseau

- **Développement** : `mlflow ui` (localhost uniquement)
- **Production** : `mlflow ui --host 0.0.0.0` (accessible réseau)

### Authentification

- **Par défaut** : Aucune authentification
- **Production** : Configurer un reverse proxy avec authentification

## Checklist de validation

- [x] MLflow installé et fonctionnel
- [x] Dossier mlruns présent avec données
- [x] Interface web accessible
- [x] Scripts de lancement créés
- [x] Documentation complète
- [x] Métriques et paramètres trackés
- [x] Rapport d'état généré

## Prochaines étapes

1. **Optimisation** : Configurer des métriques business spécifiques
2. **Automatisation** : Intégrer MLflow dans le pipeline CI/CD
3. **Monitoring** : Alertes sur les performances des modèles
4. **Registry** : Gestion des versions de modèles en production

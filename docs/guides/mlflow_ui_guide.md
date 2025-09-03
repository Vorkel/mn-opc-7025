# Guide MLflow UI - Interface de Visualisation

## Vue d'ensemble

MLflow UI est l'interface web qui permet de visualiser et comparer les expérimentations de machine learning. Elle offre une vue centralisée sur tous les runs, modèles et métriques.

## Accès à l'interface

### URL d'accès

- **Local** : http://localhost:5000
- **Réseau** : http://0.0.0.0:5000 (si lancé avec --host 0.0.0.0)

### Port par défaut

- **Port standard** : 5000
- **Port alternatif** : 8080 (si 5000 est occupé)

## Commandes de lancement

### Lancement basique

```bash
# Lancement simple
mlflow ui

# Lancement avec port personnalisé
mlflow ui --port 8080

# Lancement accessible depuis le réseau
mlflow ui --host 0.0.0.0 --port 5000
```

### Lancement en mode serveur (production)

```bash
# Serveur MLflow complet
mlflow server --host 0.0.0.0 --port 5000

# Avec backend store (SQLite)
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

## Fonctionnalités disponibles

### 1. Visualisation des Runs

- **Liste des expérimentations** : Vue d'ensemble de tous les runs
- **Métriques** : Graphiques des métriques au fil du temps
- **Paramètres** : Comparaison des hyperparamètres
- **Artifacts** : Fichiers générés (modèles, graphiques, etc.)

### 2. Comparaison des Modèles

- **Tableau comparatif** : Métriques côte à côte
- **Graphiques** : Visualisations des performances
- **Sélection** : Choix des runs à comparer

### 3. Registry des Modèles

- **Versions** : Gestion des versions de modèles
- **Staging** : Promotion des modèles (dev → staging → prod)
- **Métadonnées** : Informations sur les modèles

### 4. Métriques et Paramètres

- **Métriques** : Accuracy, AUC, Business Score, etc.
- **Paramètres** : Hyperparamètres des modèles
- **Tags** : Métadonnées personnalisées

## Configuration du projet

### Structure des dossiers

```
mlruns/
├── 0/                    # Expérimentation par défaut
│   ├── meta.yaml        # Métadonnées
│   └── [run_id]/        # Runs individuels
│       ├── meta.yaml    # Métadonnées du run
│       ├── metrics/     # Métriques
│       ├── params/      # Paramètres
│       └── artifacts/   # Fichiers générés
```

### Fichiers de configuration

- **mlruns/** : Dossier des runs (créé automatiquement)
- **mlflow.db** : Base de données SQLite (optionnel)

## Utilisation dans notre projet

### Métriques trackées

- **Business Score** : Score métier personnalisé (coût optimisé)
- **AUC Score** : Performance du modèle (0.736 pour le modèle final)
- **Model Type** : Random Forest retenu pour production
- **Feature Importance** : Variables les plus importantes identifiées

### Paramètres trackés

- **Model Type** : Type de modèle (Random Forest, LightGBM, etc.)
- **Sampling Method** : Méthode de gestion du déséquilibre
- **GridSearch Parameters** : Paramètres de recherche

### Artifacts générés

- **Modèles** : Modèles entraînés (.pkl)
- **Graphiques** : Visualisations (.png)
- **Rapports** : Rapports d'analyse (.html)

## Dépannage

### Problèmes courants

#### Port déjà utilisé

```bash
# Vérifier les processus sur le port 5000
lsof -i :5000

# Tuer le processus si nécessaire
kill -9 [PID]

# Ou utiliser un autre port
mlflow ui --port 8080
```

#### Dossier mlruns manquant

```bash
# Créer le dossier si nécessaire
mkdir -p mlruns

# Vérifier les permissions
ls -la mlruns/
```

#### Erreur de connexion

```bash
# Vérifier que MLflow est installé
pip list | grep mlflow

# Réinstaller si nécessaire
pip install mlflow
```

## Scripts utiles

### Script de lancement automatique

```bash
#!/bin/bash
echo "Lancement de MLflow UI..."

# Vérifier l'installation
if ! command -v mlflow &> /dev/null; then
    echo "MLflow non installé"
    exit 1
fi

# Créer le dossier mlruns si nécessaire
if [ ! -d "mlruns" ]; then
    echo "Création du dossier mlruns..."
    mkdir -p mlruns
fi

# Lancer l'interface
echo "Interface disponible sur : http://localhost:5000"
mlflow ui --host 0.0.0.0 --port 5000
```

## Sécurité

### Accès réseau

- **Local uniquement** : `mlflow ui` (localhost uniquement)
- **Réseau** : `mlflow ui --host 0.0.0.0` (accessible depuis le réseau)

### Authentification

- **Par défaut** : Aucune authentification
- **Production** : Configurer un reverse proxy avec authentification

## Ressources supplémentaires

- [Documentation officielle MLflow](https://mlflow.org/docs/latest/index.html)
- [Guide des métriques](https://mlflow.org/docs/latest/tracking.html#tracking-metrics)
- [API Python](https://mlflow.org/docs/latest/python_api/index.html)

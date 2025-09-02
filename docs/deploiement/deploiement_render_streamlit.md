# Guide de Déploiement : Render.com + Streamlit Cloud

## Vue d'ensemble

Ce guide vous accompagne pour déployer votre application de scoring crédit sur deux plateformes gratuites :

- **Render.com** : Pour l'API FastAPI (backend)
- **Streamlit Cloud** : Pour l'interface utilisateur (frontend)

## Prérequis

### Comptes requis

- ✅ Compte GitHub avec votre code
- ✅ Compte [Render.com](https://render.com/) (gratuit)
- ✅ Compte [Streamlit Cloud](https://share.streamlit.io/) (gratuit)

### Structure du projet

```
credit-scoring-mlops/
├── api/                    # API FastAPI
│   ├── app.py
│   └── security.py
├── streamlit_app/          # Interface Streamlit
│   ├── main.py
│   └── feature_mapping.py
├── models/                 # Modèles ML
│   ├── best_credit_model.pkl
│   └── label_encoders.pkl
├── data/                   # Données
│   └── processed/
└── requirements_minimal.txt
```

## Déploiement de l'API sur Render.com

### Étape 1 : Préparation du projet

#### 1.1 Créer le fichier `render.yaml` à la racine

```yaml
services:
  - type: web
    name: mn-opc-7025
    env: python
    plan: free
    buildCommand: pip install -r requirements_minimal.txt
    startCommand: uvicorn api.app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11
      - key: PORT
        value: 8000
    healthCheckPath: /health
    autoDeploy: true
```

#### 1.2 Optimiser `requirements_minimal.txt`

```txt
# === API CORE ===
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# === ML CORE ===
scikit-learn==1.3.0
pandas==2.3.2
numpy==1.26.4
joblib==1.5.1

# === SÉCURITÉ ===
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# === UTILITAIRES ===
python-dotenv==1.0.0
psutil==5.9.0
```

#### 1.3 Ajouter le health check dans `api/app.py`

```python
# Ajouter cette route à la fin de votre app.py
@app.get("/health")
async def health_check():
    """Health check pour Render.com"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "mn-opc-7025-api"
    }
```

#### 1.4 Vérifier la configuration de l'API

- ✅ L'API démarre avec `uvicorn api.app:app --host 0.0.0.0 --port 8000`
- ✅ Le health check `/health` répond correctement
- ✅ Les modèles ML sont accessibles depuis `models/`

### Étape 2 : Déploiement sur Render.com

#### 2.1 Connexion à Render.com

1. Allez sur [render.com](https://render.com/)
2. Cliquez sur "Sign Up" ou connectez-vous
3. Choisissez "Continue with GitHub"

#### 2.2 Créer le service web

1. Cliquez sur **"New"** → **"Web Service"**
2. Connectez votre dépôt GitHub
3. Sélectionnez le dépôt `mn-opc-7025`

#### 2.3 Configuration du service

- **Name** : `mn-opc-7025`
- **Environment** : `Python 3`
- **Region** : Choisissez la plus proche (ex: Frankfurt)
- **Branch** : `main`
- **Root Directory** : Laissez vide (racine du projet)
- **Build Command** : `pip install -r requirements_minimal.txt`
- **Start Command** : `uvicorn api.app:app --host 0.0.0.0 --port $PORT`

#### 2.4 Variables d'environnement

Ajoutez ces variables :

- **Key** : `PORT` | **Value** : `8000`
- **Key** : `PYTHON_VERSION` | **Value** : `3.11`

#### 2.5 Lancer le déploiement

1. Cliquez sur **"Create Web Service"**
2. Attendez la construction (5-10 minutes)
3. Notez l'URL : `https://mn-opc-7025.onrender.com`

#### 2.6 Vérification du déploiement

1. Testez l'URL de votre API
2. Vérifiez le health check : `/health`
3. Testez un endpoint de prédiction

## Déploiement de l'Interface sur Streamlit Cloud

### Étape 1 : Préparation du projet

#### 1.1 Créer `streamlit_app/requirements.txt`

```txt
# === INTERFACE ===
streamlit==1.28.1

# === ML CORE ===
scikit-learn==1.3.0
pandas==2.3.2
numpy==1.26.4
joblib==1.5.1

# === VISUALISATION ===
plotly==5.17.0
matplotlib==3.10.5
seaborn==0.12.2

# === UTILITAIRES ===
python-dotenv==1.0.0
```

#### 1.2 Créer `streamlit_app/.streamlit/config.toml`

```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#007bff"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

#### 1.3 Modifier `streamlit_app/main.py` pour l'API distante

```python
# Remplacer l'URL locale par l'URL Render
API_BASE_URL = "https://mn-opc-7025.onrender.com"  # Votre URL Render

# Exemple de fonction d'appel API
def call_api_prediction(data):
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=data,
            timeout=30
        )
        return response.json()
    except Exception as e:
        st.error(f"Erreur API : {str(e)}")
        return None
```

### Étape 2 : Déploiement sur Streamlit Cloud

#### 2.1 Connexion à Streamlit Cloud

1. Allez sur [share.streamlit.io](https://share.streamlit.io/)
2. Cliquez sur "Sign in with GitHub"
3. Autorisez l'accès à votre compte

#### 2.2 Créer une nouvelle application

1. Cliquez sur **"New app"**
2. Sélectionnez votre dépôt `credit-scoring-mlops`
3. Choisissez la branche `main`

#### 2.3 Configuration de l'application

- **Main file path** : `streamlit_app/main.py`
- **App URL** : Laissez la valeur par défaut
- **Advanced settings** : Laissez par défaut

#### 2.4 Lancer le déploiement

1. Cliquez sur **"Deploy!"**
2. Attendez la construction (2-5 minutes)
3. Notez l'URL : `https://votre-app-xxxx.streamlit.app`

#### 2.5 Vérification du déploiement

1. Testez l'interface utilisateur
2. Vérifiez la connexion avec l'API
3. Testez une prédiction complète

## Intégration et Tests

### Étape 1 : Vérification de la communication

#### 1.1 Test de l'API depuis Streamlit

```python
# Dans streamlit_app/main.py, ajoutez un test de connexion
def test_api_connection():
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            st.success("✅ API connectée avec succès")
            return True
        else:
            st.error("❌ Erreur de connexion API")
            return False
    except Exception as e:
        st.error(f"❌ Impossible de joindre l'API : {str(e)}")
        return False
```

#### 1.2 Test de prédiction

1. Remplissez le formulaire dans Streamlit
2. Cliquez sur "Prédire"
3. Vérifiez que la réponse arrive de l'API

### Étape 2 : Gestion des erreurs

#### 2.1 Timeout API

```python
# Dans streamlit_app/main.py
TIMEOUT_SECONDS = 30

def call_api_with_timeout(data):
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=data,
            timeout=TIMEOUT_SECONDS
        )
        return response.json()
    except requests.exceptions.Timeout:
        st.error("⏰ L'API met trop de temps à répondre")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Impossible de joindre l'API")
        return None
    except Exception as e:
        st.error(f"❌ Erreur inattendue : {str(e)}")
        return None
```

## Partie 4 : Gestion des Mises à Jour

### Étape 1 : Workflow de mise à jour

#### 1.1 Modifier le code localement

1. Faites vos modifications
2. Testez localement
3. Committez vos changements

#### 1.2 Pousser sur GitHub

```bash
git add .
git commit -m "Description des modifications"
git push origin main
```

#### 1.3 Déploiement automatique

- **Render.com** : Redéploie automatiquement en 2-5 minutes
- **Streamlit Cloud** : Redéploie automatiquement en 1-3 minutes

### Étape 2 : Monitoring

#### 2.1 Render.com

- Vérifiez les logs dans l'onglet "Logs"
- Surveillez l'utilisation dans "Metrics"
- Vérifiez le statut dans "Events"

#### 2.2 Streamlit Cloud

- Vérifiez les logs dans l'onglet "Logs"
- Surveillez les erreurs dans "Errors"
- Vérifiez les performances

## Partie 5 : Automatisation avec GitHub Actions

### Étape 1 : Configuration des Workflows CI/CD

#### 1.1 Créer le dossier `.github/workflows/`

```bash
mkdir -p .github/workflows
```

#### 1.2 Workflow de test et validation de l'API

Créez le fichier `.github/workflows/test-api.yml` :

```yaml
name: Test et Validation de l'API

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-api:
    runs-on: ubuntu-latest

    steps:
      - name: Récupération du code
        uses: actions/checkout@v4

      - name: Configuration de Python 3.11
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Installation des dépendances
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements_minimal.txt
          pip install pytest pytest-asyncio httpx

      - name: Tests de l'API
        run: |
          python -m pytest tests/api/ -v

      - name: Validation de la syntaxe
        run: |
          python -m py_compile api/app.py
          python -c "import api.app; print('✅ API importée avec succès')"

      - name: Test de démarrage de l'API
        run: |
          timeout 30s uvicorn api.app:app --host 0.0.0.0 --port 8000 &
          sleep 10
          curl -f http://localhost:8000/health || exit 1
```

#### 1.3 Workflow de test et validation de Streamlit

Créez le fichier `.github/workflows/test-streamlit.yml` :

```yaml
name: Test et Validation de Streamlit

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-streamlit:
    runs-on: ubuntu-latest

    steps:
      - name: Récupération du code
        uses: actions/checkout@v4

      - name: Configuration de Python 3.11
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Installation des dépendances Streamlit
        run: |
          python -m pip install --upgrade pip
          pip install -r streamlit_app/requirements.txt

      - name: Validation de la syntaxe Streamlit
        run: |
          python -m py_compile streamlit_app/main.py
          python -c "import streamlit_app.main; print('✅ Streamlit importé avec succès')"

      - name: Test de démarrage Streamlit (headless)
        run: |
          timeout 30s streamlit run streamlit_app/main.py --server.headless true --server.port 8501 &
          sleep 10
          curl -f http://localhost:8501 || exit 1
```

#### 1.4 Workflow de déploiement automatique sur Render

Créez le fichier `.github/workflows/deploy-render.yml` :

```yaml
name: Déploiement Automatique sur Render

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy-render:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - name: Récupération du code
        uses: actions/checkout@v4

      - name: Configuration de Python 3.11
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Tests avant déploiement
        run: |
          pip install -r requirements_minimal.txt
          pip install pytest pytest-asyncio httpx
          python -m pytest tests/api/ -v

      - name: Déploiement sur Render (via webhook)
        env:
          RENDER_WEBHOOK_URL: ${{ secrets.RENDER_WEBHOOK_URL }}
        run: |
          if [ -n "$RENDER_WEBHOOK_URL" ]; then
            echo "🚀 Déclenchement du déploiement sur Render..."
            curl -X POST "$RENDER_WEBHOOK_URL" \
              -H "Content-Type: application/json" \
              -d '{"ref": "'${{ github.ref }}'", "sha": "'${{ github.sha }}'"}'
            echo "✅ Déploiement déclenché"
          else
            echo "⚠️ RENDER_WEBHOOK_URL non configuré, déploiement manuel requis"
          fi

      - name: Notification de déploiement
        if: always()
        run: |
          if [ "${{ job.status }}" == "success" ]; then
            echo "✅ Déploiement réussi sur Render"
          else
            echo "❌ Échec du déploiement sur Render"
          fi
```

#### 1.5 Workflow de validation complète

Créez le fichier `.github/workflows/validate-all.yml` :

```yaml
name: Validation Complète du Projet

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 2 * * 1" # Tous les lundis à 2h du matin

jobs:
  validate-all:
    runs-on: ubuntu-latest

    steps:
      - name: Récupération du code
        uses: actions/checkout@v4

      - name: Configuration de Python 3.11
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Installation des dépendances
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements_minimal.txt
          pip install -r streamlit_app/requirements.txt
          pip install pytest pytest-asyncio httpx black flake8

      - name: Vérification du formatage du code
        run: |
          black --check --diff api/ streamlit_app/ src/

      - name: Vérification de la qualité du code
        run: |
          flake8 api/ streamlit_app/ src/ --max-line-length=88 --extend-ignore=E203,W503

      - name: Tests unitaires
        run: |
          python -m pytest tests/ -v --cov=src --cov=api --cov-report=xml

      - name: Validation des modèles ML
        run: |
          python -c "
          import joblib
          import os
          models_dir = 'models'
          if os.path.exists(models_dir):
              for file in os.listdir(models_dir):
                  if file.endswith('.pkl'):
                      model_path = os.path.join(models_dir, file)
                      try:
                          model = joblib.load(model_path)
                          print(f'✅ Modèle {file} chargé avec succès')
                      except Exception as e:
                          print(f'❌ Erreur chargement {file}: {e}')
                          exit(1)
          else:
              print('⚠️ Dossier models/ non trouvé')
          "

      - name: Upload des résultats de couverture
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
```

### Étape 2 : Configuration des Secrets GitHub

#### 2.1 Secrets requis pour Render.com

1. Allez dans votre dépôt GitHub → **Settings** → **Secrets and variables** → **Actions**
2. Ajoutez ces secrets :

**RENDER_WEBHOOK_URL** (optionnel)

- **Nom** : `RENDER_WEBHOOK_URL`
- **Valeur** : URL du webhook Render (à configurer dans Render.com)

#### 2.2 Configuration du webhook Render (optionnel)

1. Dans Render.com, allez dans votre service
2. **Settings** → **Webhooks**
3. Ajoutez un webhook pointant vers votre GitHub Actions

### Étape 3 : Workflows de Maintenance

#### 3.1 Workflow de nettoyage et optimisation

Créez le fichier `.github/workflows/cleanup.yml` :

```yaml
name: Nettoyage et Optimisation

on:
  schedule:
    - cron: "0 3 * * 0" # Tous les dimanches à 3h du matin
  workflow_dispatch:

jobs:
  cleanup:
    runs-on: ubuntu-latest

    steps:
      - name: Nettoyage des branches obsolètes
        run: |
          echo "Nettoyage des branches obsolètes..."
          # Logique de nettoyage des branches

      - name: Vérification de l'espace disque
        run: |
          echo "Vérification de l'espace disque..."
          df -h

      - name: Nettoyage du cache pip
        run: |
          echo "Nettoyage du cache pip..."
          pip cache purge
```

### Étape 4 : Intégration avec les Plateformes

#### 4.1 Render.com - Déploiement automatique

- ✅ Connectez votre dépôt GitHub
- ✅ Activez "Auto-Deploy" dans les paramètres
- ✅ Le déploiement se déclenche automatiquement à chaque push

#### 4.2 Streamlit Cloud - Déploiement automatique

- ✅ Connectez votre dépôt GitHub
- ✅ Le déploiement se déclenche automatiquement à chaque push
- ✅ Aucune configuration supplémentaire requise

### Étape 5 : Monitoring des Workflows

#### 5.1 Vérification des statuts

- Allez dans **Actions** de votre dépôt GitHub
- Surveillez l'exécution des workflows
- Vérifiez les logs en cas d'échec

#### 5.2 Notifications

- Configurez les notifications GitHub pour les échecs
- Intégrez avec Slack/Discord si nécessaire
- Surveillez les emails de statut

## Dépannage

### Problèmes courants Render.com

#### 1. Build échoue

- ✅ Vérifiez `requirements_minimal.txt`
- ✅ Vérifiez la syntaxe Python
- ✅ Vérifiez les imports dans `app.py`

#### 2. API ne démarre pas

- ✅ Vérifiez la commande de démarrage
- ✅ Vérifiez le port dans les variables d'environnement
- ✅ Vérifiez les logs de démarrage

#### 3. Timeout des requêtes

- ✅ Augmentez le timeout dans Streamlit
- ✅ Vérifiez la complexité des modèles ML
- ✅ Optimisez le code de prédiction

### Problèmes courants Streamlit Cloud

#### 1. Dépendances manquantes

- ✅ Vérifiez `streamlit_app/requirements.txt`
- ✅ Vérifiez les versions des packages
- ✅ Vérifiez les imports dans `main.py`

#### 2. Erreur de connexion API

- ✅ Vérifiez l'URL de l'API
- ✅ Vérifiez que l'API est accessible
- ✅ Testez avec un navigateur

## Vérification finale

### Checklist de déploiement

- [ ] API déployée sur Render.com et accessible
- [ ] Interface déployée sur Streamlit Cloud et accessible
- [ ] Communication API-Interface fonctionnelle
- [ ] Prédictions fonctionnelles
- [ ] Gestion d'erreurs en place
- [ ] Monitoring configuré

### Tests de validation

1. **Test de santé** : `/health` répond correctement
2. **Test de prédiction** : Formulaire → API → Résultat
3. **Test d'erreur** : Gestion des timeouts et erreurs
4. **Test de performance** : Temps de réponse acceptable

## Liens utiles

- [Documentation Render.com](https://render.com/docs)
- [Documentation Streamlit Cloud](https://docs.streamlit.io/streamlit-community-cloud)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Support

En cas de problème :

1. Vérifiez les logs des plateformes
2. Consultez la documentation officielle
3. Testez localement pour isoler le problème
4. Vérifiez la configuration des fichiers

---

**Note** : Ce guide est optimisé pour des applications de taille moyenne. Pour des applications très volumineuses, considérez des solutions payantes ou des optimisations supplémentaires.

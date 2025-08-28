# Configuration Hugging Face Spaces

## URLs des Spaces

### API FastAPI

- **URL** : https://kodezar-opcmne7p-api.hf.space
- **Type** : Docker
- **Branch** : main
- **Endpoints** :
  - `GET /` : Health check
  - `POST /score` : Scoring crédit

### UI Streamlit

- **URL** : https://kodezar-opcmnep7-ui.hf.space
- **Type** : Docker
- **Branch** : main
- **Variable d'environnement** : `API_URL`

## Configuration des Secrets

### GitHub Repository Secrets

1. `HF_TOKEN_API` : Token pour le Space API
2. `HF_TOKEN_UI` : Token pour le Space UI

### Space UI Variables

1. `API_URL` : `https://kodezar-opcmne7p-api.hf.space`

## Workflows CI/CD

### Déploiement API

- **Trigger** : Push sur `api/**`
- **Workflow** : `.github/workflows/deploy-hf-api.yml`
- **Action** : Build Docker + Push vers Space API

### Déploiement UI

- **Trigger** : Push sur `streamlit_app/**`
- **Workflow** : `.github/workflows/deploy-hf-ui.yml`
- **Action** : Build Docker + Push vers Space UI

## Test de Déploiement

```bash
# Test local
python test_hf_deployment.py

# Test manuel API
curl https://kodezar-opcmne7p-api.hf.space/

# Test manuel UI
curl https://kodezar-opcmnep7-ui.hf.space/
```

## Debug

### Logs API

- Space API → Settings → Logs

### Logs UI

- Space UI → Settings → Logs

### CORS

- Vérifier que l'URL du Space UI est dans `allow_origins` de l'API
- URL actuelle : `https://kodezar-opcmnep7-ui.hf.space`

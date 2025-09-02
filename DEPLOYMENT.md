# 🚀 Guide de Déploiement - Credit Scoring MLOps

## 📁 Structure des Requirements

### **1. `requirements.txt` (racine)**

- **Usage** : Développement local, tests, notebooks
- **Python** : 3.11+
- **Contenu** : Toutes les dépendances complètes
- **Déploiement** : Local uniquement

### **2. `requirements_render.txt` (racine)**

- **Usage** : Template pour Render.com (API FastAPI)
- **Python** : 3.11
- **Contenu** : Dépendances minimales API (sans conflits)
- **Déploiement** : Copié dans requirements.txt pour Render.com

### **3. `streamlit_app/requirements.txt`**

- **Usage** : Streamlit Cloud (interface utilisateur)
- **Python** : 3.13.6 (détecté automatiquement)
- **Contenu** : Dépendances minimales UI
- **Déploiement** : Automatique sur Streamlit Cloud

### **4. `streamlit_app/.streamlit/requirements.txt`**

- **Usage** : Prioritaire pour Streamlit Cloud
- **Python** : 3.13.6
- **Contenu** : Même que streamlit_app/requirements.txt

## 🌐 Déploiement

### **Render.com (API)**

```bash
# Le fichier requirements.txt (copie de requirements_render.txt) est utilisé
# Python 3.11 forcé via runtime.txt
```

### **Streamlit Cloud (UI)**

```bash
# Le fichier streamlit_app/requirements.txt est utilisé
# Python 3.13.6 détecté automatiquement
```

### **Développement Local**

```bash
# Utiliser requirements.txt (complet)
pip install -r requirements.txt
```

## 🔧 Configuration

### **Render.com**

- **Runtime** : Python 3.11
- **Requirements** : `requirements.txt` (copie de requirements_render.txt)
- **Health Check** : `/health`

### **Streamlit Cloud**

- **Runtime** : Python 3.13.6
- **Requirements** : `streamlit_app/requirements.txt`
- **API** : Endpoint public `/predict_public`

## 📋 Commandes de Déploiement

### **1. Commiter les changements**

```bash
git add .
git commit -m "Update: Structure des requirements pour déploiement"
git push origin main
```

### **2. Redémarrer Render.com**

1. Dashboard Render.com → Service API → Manual Deploy

### **3. Redémarrer Streamlit Cloud**

1. Share.streamlit.io → App → Redeploy

## ✅ Vérification

### **API Render.com**

```bash
curl https://mn-opc-7025.onrender.com/health
```

### **Streamlit Cloud**

- Vérifier les logs de déploiement
- Tester une prédiction individuelle

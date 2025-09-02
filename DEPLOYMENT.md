# 🚀 Guide de Déploiement - Credit Scoring MLOps

## 📁 Structure des Requirements

### **1. `requirements.txt` (racine)**

- **Usage** : Développement local, tests, notebooks
- **Python** : 3.11+
- **Contenu** : Toutes les dépendances complètes
- **Déploiement** : Local uniquement

### **2. `requirements_render.txt` (racine)**

- **Usage** : Render.com (API FastAPI)
- **Python** : 3.11
- **Contenu** : Dépendances minimales API (sans conflits)
- **Déploiement** : Spécifiquement pour Render.com

### **3. `streamlit_app/requirements.txt`**

- **Usage** : Streamlit Cloud (interface utilisateur)
- **Python** : 3.13.6 (détecté automatiquement)
- **Contenu** : Dépendances minimales UI
- **Déploiement** : Automatique sur Streamlit Cloud

### **3. `streamlit_app/requirements.txt`**

- **Usage** : Streamlit Cloud (interface utilisateur)
- **Python** : 3.13.6 (détecté automatiquement)
- **Contenu** : Dépendances minimales UI
- **Déploiement** : Automatique sur Streamlit Cloud

## 🌐 Déploiement

### **Render.com (API)**

```bash
# Le fichier requirements.txt est utilisé automatiquement
# Python 3.11 forcé via runtime.txt
```

### **Streamlit Cloud (UI)**

```bash
# Le fichier streamlit_app/requirements.txt est utilisé
# Python 3.13.6 détecté automatiquement
```

### **Développement Local**

```bash
# Utiliser requirements_dev.txt
pip install -r requirements_dev.txt
```

## 🔧 Configuration

### **Render.com**

- **Runtime** : Python 3.11
- **Requirements** : `requirements.txt`
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

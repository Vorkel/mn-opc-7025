# Configuration API Distante - Guide Complet

## 🎯 Résumé de la Configuration

### **Détection Automatique d'Environnement**

La configuration se fait automatiquement selon l'environnement :

```python
# Configuration automatique dans streamlit_app/main.py
IS_PRODUCTION = (
    os.getenv('STREAMLIT_ENV') == 'production' or
    os.getenv('RENDER') is not None
)
USE_REMOTE_API = IS_PRODUCTION
```

## 🔧 **Modes de Fonctionnement**

### **1. Mode LOCAL (Développement)**

```bash
# Lancement standard
make start-streamlit
# ou
streamlit run streamlit_app/main.py
```

**Configuration :**

- `USE_REMOTE_API = False`
- ✅ Modèle local uniquement
- ✅ Tests rapides sans dépendance réseau
- ✅ Développement offline

### **2. Mode PRODUCTION (Déployé)**

```bash
# Lancement avec API distante
make start-streamlit-prod
# ou
STREAMLIT_ENV=production streamlit run streamlit_app/main.py
```

**Configuration :**

- `USE_REMOTE_API = True`
- ✅ API Render.com en priorité
- ✅ Fallback automatique sur modèle local si API indisponible
- ✅ Scalabilité et monitoring optimaux

## 🚀 **Déploiement sur Render.com**

### **Variables d'Environnement à Configurer :**

1. **Détection automatique** : Render.com définit automatiquement `RENDER=1`
2. **Force production** (optionnel) : `STREAMLIT_ENV=production`

### **Comportement en Production :**

1. **Tentative API distante** : `https://mn-opc-7025.onrender.com/health`
2. **Si succès** : Utilise l'API pour toutes les prédictions
3. **Si échec** : Bascule automatiquement sur le modèle local
4. **Fallback** : Aucune interruption de service

## ⚡ **Avantages de cette Configuration**

### **Pour le Développement :**

- ✅ Pas de latence réseau
- ✅ Tests offline
- ✅ Débogage simplifié
- ✅ Pas de dépendance externe

### **Pour la Production :**

- ✅ Performance optimale via API
- ✅ Monitoring centralisé
- ✅ Scalabilité horizontale
- ✅ Continuité de service (fallback)

## 🔍 **Vérification de la Configuration**

```bash
# Vérifier le mode actuel
python -c "
import os
IS_PRODUCTION = (os.getenv('STREAMLIT_ENV') == 'production' or os.getenv('RENDER') is not None)
print(f'Mode: {\"PRODUCTION\" if IS_PRODUCTION else \"LOCAL\"}')
print(f'API Distante: {\"ACTIVÉE\" if IS_PRODUCTION else \"DÉSACTIVÉE\"}')
"
```

## 🛠️ **Commandes Utiles**

```bash
# Mode développement
make start-streamlit

# Mode production (simulation)
make start-streamlit-prod

# Vérifier configuration
scripts/streamlit_launcher.sh

# Logs en temps réel
make logs
```

## 📊 **Monitoring**

Le système affiche automatiquement :

- ✅ Source du modèle (API/Local)
- ✅ Statut de connexion API
- ✅ Mode de fonctionnement actuel
- ✅ Métriques de fallback

## ⚠️ **Points Importants**

1. **Local d'abord** : Toujours tester en mode local avant déploiement
2. **Fallback garanti** : Le modèle local assure la continuité même si API indisponible
3. **Variables automatiques** : Render.com active automatiquement le mode production
4. **Pas de modification manuelle** : La configuration s'adapte automatiquement à l'environnement

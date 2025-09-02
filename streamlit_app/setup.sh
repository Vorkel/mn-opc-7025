#!/bin/bash
# Script de configuration pour Streamlit Cloud

echo "🚀 Configuration de l'environnement Streamlit..."

# Vérifier la version de Python
python --version

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# Vérifier l'installation
python -c "import streamlit; print(f'Streamlit version: {streamlit.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

echo "✅ Configuration terminée !"

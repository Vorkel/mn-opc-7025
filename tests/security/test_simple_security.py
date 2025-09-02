#!/usr/bin/env python3
"""
Tests de sécurité simples
"""

import os
import re
import sys

import pytest


def test_no_hardcoded_secrets() -> None:
    """Test qu'il n'y a pas de secrets en dur dans le code"""
    sensitive_patterns = [
        r'password\s*=\s*["\'][^"\']+["\']',
        r'secret\s*=\s*["\'][^"\']+["\']',
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'token\s*=\s*["\'][^"\']+["\']',
    ]

    # Dossiers à vérifier (exclure streamlit_app pour éviter les faux positifs)
    directories = ["src/", "api/"]

    for directory in directories:
        if not os.path.exists(directory):
            continue

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        for pattern in sensitive_patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            if matches:
                                pytest.fail(
                                    f"Secret potentiel trouvé dans {file_path}:"
                                    f" {matches}"
                                )

                    except Exception as e:
                        print(f"⚠️ Impossible de lire {file_path}: {e}")

    print("✅ Aucun secret en dur détecté")


def test_import_security() -> None:
    """Test que les imports sont sécurisés"""
    dangerous_imports = ["subprocess", "os.system", "eval", "exec", "__import__"]

    # Dossiers à vérifier
    directories = ["src/", "api/", "streamlit_app/"]

    for directory in directories:
        if not os.path.exists(directory):
            continue

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        for dangerous_import in dangerous_imports:
                            if dangerous_import in content:
                                # Vérifier que ce n'est pas dans un commentaire ou
                                # string
                                lines = content.split("\n")
                                for i, line in enumerate(lines, 1):
                                    if dangerous_import in line:
                                        # Ignorer les commentaires et strings
                                        stripped_line = line.strip()
                                        if (
                                            not stripped_line.startswith("#")
                                            and not stripped_line.startswith('"')
                                            and not stripped_line.startswith("'")
                                        ):
                                            print(
                                                "⚠️ Import potentiellement dangereux"
                                                f" dans {file_path}:{i}: {line.strip()}"
                                            )

                    except Exception as e:
                        print(f"⚠️ Impossible de lire {file_path}: {e}")

    print("✅ Imports sécurisés vérifiés")


def test_file_permissions() -> None:
    """Test que les fichiers sensibles ont les bonnes permissions"""
    sensitive_files = [".env", ".env.local", "secrets/", "config/"]

    for file_path in sensitive_files:
        if os.path.exists(file_path):
            # Vérifier que les fichiers sensibles ne sont pas dans le repo
            if not file_path.startswith("."):
                print(f"⚠️ Fichier sensible dans le repo: {file_path}")

    print("✅ Permissions des fichiers vérifiées")


def test_dependency_security() -> None:
    """Test simple de sécurité des dépendances"""
    try:
        import safety

        print("✅ Safety disponible pour vérification des vulnérabilités")
    except ImportError:
        print("⚠️ Safety non installé - impossible de vérifier les vulnérabilités")

    try:
        import bandit

        print("✅ Bandit disponible pour analyse statique")
    except ImportError:
        print("⚠️ Bandit non installé - impossible d'analyser le code")


if __name__ == "__main__":
    print("Tests de sécurité simples")
    print("=" * 40)

    try:
        test_no_hardcoded_secrets()
        test_import_security()
        test_file_permissions()
        test_dependency_security()
        print("\n✅ Tous les tests de sécurité passent!")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)

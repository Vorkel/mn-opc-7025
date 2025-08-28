#!/usr/bin/env python3
"""
Small helper to package and upload API/UI to Hugging Face Spaces.

Requires environment:
  HF_TOKEN           (token with write on Spaces)
  HF_SPACE_ID        (e.g. username/space-name)
  COMPONENT          ("api" or "ui")

Usage:
  python tools/push_to_hf.py
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, create_repo, upload_folder


ROOT = Path(__file__).resolve().parents[1]


def stage_api(dst: Path) -> None:
    # Prepare minimal Docker Space for FastAPI API
    # Copy api/ source
    shutil.copytree(ROOT / "api", dst / "api")
    # Include minimal requirements
    shutil.copy2(ROOT / "requirements_minimal.txt", dst / "requirements.txt")
    # Copy model if exists
    models_src = ROOT / "models" / "best_credit_model.pkl"
    if models_src.is_file():
        (dst / "models").mkdir(parents=True, exist_ok=True)
        shutil.copy2(models_src, dst / "models" / "best_credit_model.pkl")
    # Dockerfile for Spaces
    (dst / "Dockerfile").write_text(
        """
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PORT=7860 MODEL_PATH=models/best_credit_model.pkl
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY api ./api
COPY models ./models
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
""".strip()
    )
    (dst / "README.md").write_text(
        "# Credit Scoring API\n\nFastAPI app for credit scoring (Hugging Face Space).\n"
    )


def stage_ui(dst: Path) -> None:
    # Prepare Streamlit Space
    shutil.copytree(ROOT / "streamlit_app", dst / "streamlit_app")
    # Entrypoint expected by HF (Streamlit app at repo root)
    (dst / "app.py").write_text(
        """
import sys
from pathlib import Path

# Ensure 'streamlit_app' directory is importable for absolute imports inside main.py
sys.path.append(str(Path(__file__).parent / "streamlit_app"))

# Import the Streamlit app script (executes on import)
import main  # type: ignore  # noqa: F401
""".strip()
    )
    # Requirements: reuse project requirements.txt for simplicity
    shutil.copy2(ROOT / "requirements.txt", dst / "requirements.txt")
    # Copy model if exists so UI can run standalone
    models_src = ROOT / "models" / "best_credit_model.pkl"
    if models_src.is_file():
        (dst / "models").mkdir(parents=True, exist_ok=True)
        shutil.copy2(models_src, dst / "models" / "best_credit_model.pkl")
    (dst / "README.md").write_text(
        "# Credit Scoring UI\n\nStreamlit app for credit scoring (Hugging Face Space).\n"
    )


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    space_id = os.environ.get("HF_SPACE_ID", "").strip()
    component = os.environ.get("COMPONENT", "").strip().lower()
    if not token or not space_id or component not in {"api", "ui"}:
        raise SystemExit(
            "Missing env. Need HF_TOKEN, HF_SPACE_ID, COMPONENT in {api,ui}."
        )

    api = HfApi(token=token)
    # Create/update space (sdk type based on component)
    sdk = "docker" if component == "api" else "streamlit"
    create_repo(space_id, repo_type="space", exist_ok=True, space_sdk=sdk, token=token)

    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp)
        if component == "api":
            stage_api(dst)
        else:
            stage_ui(dst)
        upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path=str(dst),
            token=token,
            commit_message=f"Deploy {component} from GitHub Actions",
        )


if __name__ == "__main__":
    main()

# FineTuneFlow — Gestion d'Erreurs

## 1. Philosophie

- **Chaque erreur a un code** identifiable programmatiquement
- **Chaque erreur a un message** en anglais lisible par l'utilisateur
- **Les erreurs ne cassent pas le système** — un job qui échoue laisse le projet dans un état récupérable
- **Les erreurs sont loguées** (DB + fichier + console)
- **Les erreurs récupérables sont retryées** automatiquement

## 2. Format d'erreur API standard

```python
from pydantic import BaseModel
from typing import Optional, Any

class ErrorResponse(BaseModel):
    detail: str                          # message lisible
    error_code: str                      # code machine (ex: "FILE_TOO_LARGE")
    context: Optional[dict[str, Any]] = None  # données additionnelles
```

**Exemple réponse HTTP 400 :**
```json
{
  "detail": "File 'report.pdf' exceeds maximum size of 50 MB (actual: 67.2 MB)",
  "error_code": "FILE_TOO_LARGE",
  "context": {
    "filename": "report.pdf",
    "max_size_mb": 50,
    "actual_size_mb": 67.2
  }
}
```

## 3. Catalogue des codes d'erreur

### 3.1 Projet

| Code | HTTP | Description |
|------|------|-------------|
| `PROJECT_NOT_FOUND` | 404 | Projet avec cet ID n'existe pas |
| `PROJECT_INVALID_STATUS` | 409 | Opération incompatible avec le statut actuel |
| `PROJECT_DELETE_RUNNING` | 409 | Impossible de supprimer un projet avec des jobs en cours |

### 3.2 Fichiers

| Code | HTTP | Description |
|------|------|-------------|
| `FILE_TOO_LARGE` | 413 | Fichier dépasse la taille max |
| `FILE_TOTAL_LIMIT` | 413 | Total des fichiers du projet dépasse la limite |
| `FILE_UNSUPPORTED_TYPE` | 400 | Type MIME non supporté |
| `FILE_INVALID_NAME` | 400 | Nom de fichier invalide (path traversal, etc.) |
| `FILE_NOT_FOUND` | 404 | Fichier non trouvé |
| `FILE_UPLOAD_FAILED` | 500 | Erreur lors de l'écriture sur disque |
| `FILE_EMPTY` | 400 | Fichier vide (0 bytes) |

### 3.3 Modèle HF

| Code | HTTP | Description |
|------|------|-------------|
| `MODEL_NOT_FOUND` | 404 | Modèle introuvable sur HuggingFace Hub |
| `MODEL_ACCESS_DENIED` | 403 | Modèle gated, accès refusé |
| `MODEL_RESOLVE_FAILED` | 502 | Erreur de communication avec HF Hub |
| `MODEL_UNSUPPORTED_ARCH` | 400 | Architecture non supportée pour SFT |

### 3.4 Dataset

| Code | HTTP | Description |
|------|------|-------------|
| `DATASET_NO_CHUNKS` | 400 | Aucun chunk disponible pour la génération |
| `DATASET_NO_DOCS` | 400 | Aucun document uploadé |
| `DATASET_GENERATION_FAILED` | 500 | Génération a échoué (après retries) |
| `DATASET_TOO_FEW_EXAMPLES` | 400 | Moins de 10 exemples valides (minimum pour training) |
| `DATASET_UPLOAD_INVALID` | 400 | Dataset uploadé invalide (format, champs manquants) |

### 3.5 Ollama Cloud

| Code | HTTP | Description |
|------|------|-------------|
| `OLLAMA_API_KEY_MISSING` | 400 | Clé API Ollama non configurée |
| `OLLAMA_API_KEY_INVALID` | 401 | Clé API Ollama rejetée |
| `OLLAMA_API_TIMEOUT` | 504 | Timeout sur l'appel Ollama Cloud |
| `OLLAMA_API_RATE_LIMIT` | 429 | Rate limit Ollama Cloud atteint |
| `OLLAMA_API_ERROR` | 502 | Erreur serveur Ollama Cloud |
| `OLLAMA_MODEL_NOT_FOUND` | 404 | Modèle Ollama demandé non disponible |
| `OLLAMA_RESPONSE_INVALID` | 500 | Réponse Ollama non parseable (même après repair) |

### 3.6 Hardware

| Code | HTTP | Description |
|------|------|-------------|
| `GPU_NOT_AVAILABLE` | 200* | Pas de GPU CUDA détecté |
| `GPU_VRAM_INSUFFICIENT` | 200* | VRAM insuffisante pour le modèle choisi |
| `GPU_PROBE_FAILED` | 500 | Erreur lors du probe hardware |

> *200 car c'est un état factuel, pas une erreur de requête

### 3.7 Training

| Code | HTTP | Description |
|------|------|-------------|
| `TRAINING_ALREADY_RUNNING` | 409 | Un training est déjà en cours pour ce projet |
| `TRAINING_NO_DATASET` | 400 | Pas de dataset prêt pour le training |
| `TRAINING_NO_GPU` | 400 | Pas de GPU disponible |
| `TRAINING_OOM` | 500 | Out of memory pendant le training |
| `TRAINING_MODEL_LOAD_FAILED` | 500 | Impossible de charger le modèle |
| `TRAINING_INTERRUPTED` | 500 | Training interrompu (crash, kill) |
| `TRAINING_DIVERGED` | 500 | Loss a divergé (NaN ou Inf) |

### 3.8 Export

| Code | HTTP | Description |
|------|------|-------------|
| `EXPORT_NO_RUN` | 400 | Aucun training terminé à exporter |
| `EXPORT_ARTIFACTS_MISSING` | 500 | Fichiers d'artifacts manquants sur disque |

### 3.9 Jobs

| Code | HTTP | Description |
|------|------|-------------|
| `JOB_NOT_FOUND` | 404 | Job non trouvé |
| `JOB_ALREADY_FINISHED` | 409 | Job déjà terminé, impossible d'annuler |
| `JOB_CANCEL_FAILED` | 500 | Impossible d'annuler le job Celery |

## 4. Stratégie de retry

### 4.1 Celery tasks — Retry automatique

```python
# Configuration par type de tâche
RETRY_CONFIGS = {
    "dataset_preview": {
        "autoretry_for": (OllamaAPIError, ConnectionError, TimeoutError),
        "max_retries": 3,
        "retry_backoff": True,       # backoff exponentiel
        "retry_backoff_max": 60,     # max 60s entre retries
        "retry_jitter": True,        # jitter pour éviter thundering herd
    },
    "dataset_generate": {
        "autoretry_for": (OllamaAPIError, ConnectionError, TimeoutError),
        "max_retries": 3,
        "retry_backoff": True,
        "retry_backoff_max": 120,
    },
    "train": {
        "autoretry_for": (),  # PAS de retry auto pour training (trop coûteux)
        "max_retries": 0,
    },
    "export": {
        "autoretry_for": (IOError,),
        "max_retries": 2,
        "retry_backoff": True,
    },
}
```

### 4.2 Ollama Cloud — Retry HTTP

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
)
async def call_ollama(messages: list, **params) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
            json={"messages": messages, **params}
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
```

### 4.3 JSON Repair — Pipeline de retry

```
Étape 1: Parse strict (json.loads)           → OK? terminé
Étape 2: Strip markdown wrappers             → Retry parse → OK? terminé
Étape 3: Regex extract JSON objects           → Retry parse → OK? terminé
Étape 4: Fix common issues (trailing commas)  → Retry parse → OK? terminé
Étape 5: Appel LLM repair (1 seule fois)     → Retry parse → OK? terminé
Étape 6: REJET — marquer invalid
```

## 5. Gestion d'état des jobs en cas d'erreur

```
Job créé (queued)
  │
  ├── Worker prend le job → status = running
  │     │
  │     ├── Succès → status = success, project.status = next_state
  │     │
  │     ├── Erreur retryable → status = retrying, retry_count++
  │     │     └── Si retry_count >= max_retries → status = failed
  │     │
  │     ├── Erreur fatale → status = failed, project.status = failed?
  │     │     └── (OOM, modèle pas trouvé, etc.)
  │     │
  │     └── Annulation → status = cancelled
  │
  └── Worker ne prend jamais le job (Redis down)
        └── Timeout côté API → status = failed après X minutes
```

### 5.1 Impact sur le projet

| Job status | Project status | Commentaire |
|------------|---------------|-------------|
| `dataset_generate` → failed | Reste `generating` | L'utilisateur peut re-lancer |
| `train` → failed | Revient à `ready_to_train` | L'utilisateur peut re-lancer |
| `train` → OOM | Revient à `ready_to_train` | Suggérer QLoRA ou réduire batch |
| `export` → failed | Reste `completed` | Re-tenter l'export |

## 6. Logging

### 6.1 Niveaux

```python
import structlog

logger = structlog.get_logger()

# Utilisation
logger.info("job.started", job_id=job_id, type=job_type)
logger.warning("ollama.retry", attempt=2, error="timeout")
logger.error("training.oom", job_id=job_id, vram_used_gb=23.8)
```

### 6.2 Destinations

| Destination | Contenu | Rétention |
|-------------|---------|-----------|
| Console (stdout) | Tout | Aucune |
| Fichier `logs/{date}.log` | Tout | 7 jours |
| DB (`jobs.error_message`) | Erreurs finales | Permanent |
| Redis Pub/Sub | Logs training temps réel | Éphémère |
| DB (`jobs.result_summary`) | Résumés succès | Permanent |

## 7. Exception classes

```python
# backend/app/core/exceptions.py

class FineTuneFlowError(Exception):
    """Base exception."""
    error_code: str = "INTERNAL_ERROR"
    status_code: int = 500
    
    def __init__(self, detail: str, context: dict = None):
        self.detail = detail
        self.context = context or {}
        super().__init__(detail)

class NotFoundError(FineTuneFlowError):
    status_code = 404

class ProjectNotFoundError(NotFoundError):
    error_code = "PROJECT_NOT_FOUND"

class FileNotFoundError(NotFoundError):
    error_code = "FILE_NOT_FOUND"

class ConflictError(FineTuneFlowError):
    status_code = 409

class TrainingAlreadyRunningError(ConflictError):
    error_code = "TRAINING_ALREADY_RUNNING"

class ValidationError(FineTuneFlowError):
    error_code = "VALIDATION_ERROR"
    status_code = 400

class FileTooLargeError(FineTuneFlowError):
    error_code = "FILE_TOO_LARGE"
    status_code = 413

class OllamaAPIError(FineTuneFlowError):
    error_code = "OLLAMA_API_ERROR"
    status_code = 502

class GPUNotAvailableError(FineTuneFlowError):
    error_code = "GPU_NOT_AVAILABLE"
    status_code = 400

class TrainingOOMError(FineTuneFlowError):
    error_code = "TRAINING_OOM"
    status_code = 500
```

### 7.1 Exception handler FastAPI

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(FineTuneFlowError)
async def finetuneflow_error_handler(request: Request, exc: FineTuneFlowError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": exc.error_code,
            "context": exc.context,
        }
    )
```

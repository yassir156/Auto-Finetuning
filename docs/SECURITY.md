# FineTuneFlow — Sécurité

> App locale (pas d'auth utilisateur), mais sécurisation des entrées obligatoire.

## 1. Pas d'authentification

L'application tourne **en local sur le PC de l'utilisateur**. Il n'y a pas besoin d'authentification pour accéder aux endpoints. Le seul "secret" est la clé API Ollama Cloud, stockée dans `.env`.

## 2. Validation des uploads

### 2.1 Taille maximum

```python
# backend/app/core/config.py

class Settings(BaseSettings):
    # Upload limits
    MAX_FILE_SIZE_MB: int = 50              # par fichier
    MAX_PROJECT_TOTAL_SIZE_MB: int = 500    # total par projet
    MAX_FILES_PER_UPLOAD: int = 20          # par requête upload
    MAX_FILES_PER_PROJECT: int = 100        # total par projet
```

**Implémentation :**

```python
from fastapi import UploadFile, HTTPException

async def validate_upload_size(file: UploadFile, max_size_bytes: int):
    """Valider la taille sans lire tout en mémoire."""
    # Vérifier Content-Length header si disponible
    if file.size and file.size > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File '{file.filename}' exceeds max size of {max_size_bytes // 1_000_000} MB"
        )
    
    # Vérifier en lisant par chunks (cas où Content-Length absent)
    total = 0
    chunk_size = 1024 * 1024  # 1 MB
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_size_bytes:
            raise HTTPException(status_code=413, detail=f"File exceeds max size")
    
    await file.seek(0)  # reset pour lecture ultérieure
    return total
```

### 2.2 Types MIME autorisés

```python
ALLOWED_MIME_TYPES = {
    "raw_doc": {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "text/plain",
        "text/markdown",
    },
    "dataset_upload": {
        "application/json",
        "text/plain",           # .jsonl souvent détecté comme text/plain
        "text/csv",
        "application/csv",
    },
}

ALLOWED_EXTENSIONS = {
    "raw_doc": {".pdf", ".docx", ".txt", ".md"},
    "dataset_upload": {".jsonl", ".json", ".csv"},
}
```

**Validation double :**
1. Extension du fichier
2. Content-type (MIME) — via `python-magic` ou header

```python
import magic

def validate_mime_type(file_path: str, kind: str) -> str:
    """Détecte le vrai type MIME du fichier (pas juste l'extension)."""
    mime = magic.from_file(file_path, mime=True)
    if mime not in ALLOWED_MIME_TYPES[kind]:
        raise ValidationError(
            f"Detected MIME type '{mime}' not allowed for '{kind}'",
            error_code="FILE_UNSUPPORTED_TYPE"
        )
    return mime
```

### 2.3 Validation fichiers vides

```python
if file.size == 0 or total_bytes == 0:
    raise ValidationError("File is empty", error_code="FILE_EMPTY")
```

## 3. Sandboxing des fichiers (Path Traversal)

### 3.1 Problème

Un nom de fichier malveillant comme `../../etc/passwd` ou `../../../app/main.py` pourrait écrire en dehors du dossier storage.

### 3.2 Solution — Sanitisation stricte

```python
import os
import re
import uuid
from pathlib import Path

STORAGE_ROOT = Path("/app/storage")

def sanitize_filename(filename: str) -> str:
    """
    Sanitise un nom de fichier uploadé.
    Retire tout caractère dangereux, empêche le path traversal.
    """
    # 1. Extraire seulement le nom de base (pas de chemin)
    filename = os.path.basename(filename)
    
    # 2. Retirer les caractères non autorisés
    # Garder uniquement: lettres, chiffres, -, _, .
    filename = re.sub(r'[^\w\-.]', '_', filename)
    
    # 3. Retirer les .. (path traversal)
    filename = filename.replace('..', '_')
    
    # 4. Retirer les points en début (fichiers cachés)
    filename = filename.lstrip('.')
    
    # 5. Si vide après sanitisation, générer un nom
    if not filename:
        filename = f"file_{uuid.uuid4().hex[:8]}"
    
    # 6. Limiter la longueur
    name, ext = os.path.splitext(filename)
    if len(name) > 200:
        name = name[:200]
    filename = name + ext
    
    return filename


def safe_storage_path(project_id: str, subdir: str, filename: str) -> Path:
    """
    Construit un chemin de stockage sûr et vérifie qu'il reste dans STORAGE_ROOT.
    """
    sanitized = sanitize_filename(filename)
    
    # Construire le chemin
    target = STORAGE_ROOT / project_id / subdir / sanitized
    
    # RÉSOLUTION et vérification qu'on reste dans STORAGE_ROOT
    resolved = target.resolve()
    storage_resolved = STORAGE_ROOT.resolve()
    
    if not str(resolved).startswith(str(storage_resolved)):
        raise ValidationError(
            f"Path traversal detected for filename '{filename}'",
            error_code="FILE_INVALID_NAME"
        )
    
    return target
```

### 3.3 Tests de path traversal

```python
# Ces cas doivent être BLOQUÉS :
assert sanitize_filename("../../etc/passwd") == "etc_passwd"
assert sanitize_filename("../../../app/main.py") == "app_main.py"
assert sanitize_filename("/etc/shadow") == "etc_shadow"
assert sanitize_filename("..\\..\\windows\\system32") == "windows_system32"
assert sanitize_filename("file\x00name.pdf") == "file_name.pdf"

# Ces cas doivent être ACCEPTÉS :
assert sanitize_filename("rapport-final.pdf") == "rapport-final.pdf"
assert sanitize_filename("data_v2.jsonl") == "data_v2.jsonl"
assert sanitize_filename("mon document (1).pdf") == "mon_document__1_.pdf"
```

## 4. Rate Limiting

### 4.1 Outil : slowapi

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

### 4.2 Limites par endpoint

```python
# Upload — protéger le disque
@router.post("/projects/{project_id}/files/upload")
@limiter.limit("10/minute")
async def upload_files(...):
    ...

# Génération dataset — protéger l'API Ollama Cloud (coûteux)
@router.post("/projects/{project_id}/dataset/preview")
@limiter.limit("5/minute")
async def dataset_preview(...):
    ...

@router.post("/projects/{project_id}/dataset/generate")
@limiter.limit("3/minute")
async def dataset_generate(...):
    ...

# Training — protéger le GPU
@router.post("/projects/{project_id}/train/start")
@limiter.limit("3/minute")
async def train_start(...):
    ...

# Read endpoints — plus permissif
@router.get("/projects")
@limiter.limit("60/minute")
async def list_projects(...):
    ...
```

### 4.3 Réponse rate limit

```
HTTP 429 Too Many Requests

{
  "detail": "Rate limit exceeded: 10 per 1 minute",
  "error_code": "RATE_LIMIT_EXCEEDED"
}
```

## 5. Sécurité de la clé API Ollama

### 5.1 Stockage

- Stockée dans `.env` (pas commité dans git)
- Jamais loguée en texte clair
- Jamais exposée dans les réponses API
- `.env.example` contient un placeholder `CHANGE_ME`

### 5.2 Validation au démarrage

```python
@app.on_event("startup")
async def validate_config():
    if not settings.OLLAMA_CLOUD_API_KEY or settings.OLLAMA_CLOUD_API_KEY == "CHANGE_ME":
        logger.warning(
            "Ollama Cloud API key not configured. "
            "Dataset generation will not work. "
            "Set OLLAMA_CLOUD_API_KEY in .env"
        )
```

### 5.3 Masquage dans les logs

```python
def mask_api_key(key: str) -> str:
    """Masque la clé API pour les logs."""
    if not key or len(key) < 8:
        return "***"
    return key[:4] + "..." + key[-4:]
```

## 6. Protection contre les abus (même en local)

### 6.1 Limites de ressources

| Ressource | Limite | Raison |
|-----------|--------|--------|
| Taille fichier | 50 MB | Éviter de remplir le disque |
| Total fichiers projet | 500 MB | Idem |
| Nombre de chunks | 10 000 par projet | Éviter boucle infinie de chunking |
| Nombre d'exemples | 50 000 par projet | Éviter explosion de la DB |
| Temps job max | 24h (train), 2h (dataset) | Celery task time limit |
| Nombre jobs simultanés | 1 (gpu_queue), 3 (default) | Ressources machine |

### 6.2 Celery task time limits

```python
@celery_app.task(
    time_limit=7200,      # hard kill après 2h
    soft_time_limit=7000,  # graceful stop après ~1h57
)
def dataset_generate_task(...):
    ...

@celery_app.task(
    time_limit=86400,      # hard kill après 24h
    soft_time_limit=85800, # graceful stop
)
def train_task(...):
    ...
```

## 7. Checklist sécurité

- [ ] Upload : taille max validée côté serveur (pas juste frontend)
- [ ] Upload : MIME type vérifié avec `python-magic` (pas juste l'extension)
- [ ] Upload : nom de fichier sanitisé
- [ ] Upload : path résolu vérifié dans STORAGE_ROOT
- [ ] API : rate limiting sur tous les endpoints d'écriture
- [ ] API : CORS configuré (localhost uniquement)
- [ ] Env : `.env` dans `.gitignore`
- [ ] Env : clé API jamais loguée en clair
- [ ] Jobs : time limits sur toutes les tâches Celery
- [ ] Jobs : concurrency=1 sur gpu_queue
- [ ] DB : paramètres SQL via ORM (pas de raw SQL avec interpolation)
- [ ] Storage : pas de fichiers servis directement (via API seulement)

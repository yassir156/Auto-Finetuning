# FineTuneFlow — Schéma Base de Données (PostgreSQL 16)

## 1. Types Enum

```sql
-- Task types supportés (MVP)
CREATE TYPE task_type AS ENUM ('instruction', 'qa');

-- Statut global d'un projet
CREATE TYPE project_status AS ENUM (
  'draft',           -- projet créé, pas encore de données
  'uploading',       -- upload en cours
  'chunking',        -- chunking docs en cours
  'generating',      -- génération dataset en cours
  'ready_to_train',  -- dataset prêt, en attente de training
  'training',        -- fine-tuning en cours
  'evaluating',      -- évaluation en cours
  'completed',       -- tout terminé, artifacts disponibles
  'failed'           -- erreur non récupérable
);

-- Type de fichier stocké
CREATE TYPE file_kind AS ENUM (
  'raw_doc',            -- document uploadé (PDF, DOCX, TXT)
  'dataset_upload',     -- dataset uploadé directement (JSONL, CSV)
  'dataset_generated',  -- dataset généré (train.jsonl, eval.jsonl)
  'artifact',           -- adapter, config, tokenizer
  'export',             -- zip export final
  'log'                 -- fichier de log training
);

-- Statut d'un fichier
CREATE TYPE file_status AS ENUM ('uploading', 'ready', 'processing', 'failed');

-- Type de job
CREATE TYPE job_type AS ENUM (
  'chunking',           -- extraction + chunking des documents
  'dataset_preview',    -- génération preview (10 exemples)
  'dataset_generate',   -- génération complète du dataset
  'train',              -- fine-tuning
  'eval',               -- évaluation
  'export'              -- export artifacts
);

-- Statut d'un job
CREATE TYPE job_status AS ENUM (
  'queued',    -- en attente dans la queue Celery
  'running',   -- en cours d'exécution
  'success',   -- terminé avec succès
  'failed',    -- échoué
  'retrying',  -- en cours de retry après erreur
  'cancelled'  -- annulé par l'utilisateur
);

-- Méthode de fine-tuning
CREATE TYPE finetune_method AS ENUM ('lora', 'qlora');

-- Split du dataset
CREATE TYPE dataset_split AS ENUM ('preview', 'train', 'eval');
```

## 2. Tables

### 2.1 `projects`

Table principale. Un projet = un workflow complet de fine-tuning.

```sql
CREATE TABLE projects (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name          VARCHAR(255) NOT NULL,
  description   TEXT,
  task_type     task_type NOT NULL DEFAULT 'instruction',
  base_model_id VARCHAR(255),          -- HF model id (ex: "meta-llama/Llama-3.1-8B")
  model_info    JSONB,                 -- metadata HF résolue (params, arch, size, etc.)
  status        project_status NOT NULL DEFAULT 'draft',
  config        JSONB NOT NULL DEFAULT '{}',  -- config utilisateur (nb exemples, max_length, etc.)
  error_message TEXT,                  -- dernière erreur si status=failed
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_projects_created ON projects(created_at DESC);
```

**Champ `config` (JSONB) — structure typique :**
```json
{
  "num_examples_target": 2000,
  "max_seq_length": 2048,
  "train_eval_split": 0.9,
  "ollama_model": "llama3.1:70b",
  "examples_per_chunk": 5,
  "chunk_size_tokens": 512,
  "chunk_overlap_tokens": 50
}
```

**Champ `model_info` (JSONB) — structure typique :**
```json
{
  "model_id": "meta-llama/Llama-3.1-8B",
  "model_type": "llama",
  "num_parameters": 8000000000,
  "estimated_vram_fp16_gb": 16.0,
  "estimated_vram_4bit_gb": 5.5,
  "architecture": "LlamaForCausalLM",
  "vocab_size": 128256,
  "max_position_embeddings": 131072,
  "license": "llama3.1"
}
```

### 2.2 `files`

Tout fichier physique sur disque lié à un projet.

```sql
CREATE TABLE files (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id    UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind          file_kind NOT NULL,
  status        file_status NOT NULL DEFAULT 'ready',
  filename      VARCHAR(512) NOT NULL,   -- nom original du fichier
  mime_type     VARCHAR(128),
  storage_path  VARCHAR(1024) NOT NULL,  -- chemin relatif dans ./storage/
  size_bytes    BIGINT NOT NULL DEFAULT 0,
  metadata      JSONB DEFAULT '{}',      -- metadata libre (pages, encoding, etc.)
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_files_project ON files(project_id);
CREATE INDEX idx_files_kind ON files(project_id, kind);
```

### 2.3 `chunks`

Segments de texte extraits des documents. Table dédiée (pas un fichier).

```sql
CREATE TABLE chunks (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id    UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  source_file_id UUID NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_index   INTEGER NOT NULL,         -- ordre dans le document source
  content       TEXT NOT NULL,            -- texte du chunk
  token_count   INTEGER NOT NULL DEFAULT 0,
  char_count    INTEGER NOT NULL DEFAULT 0,
  metadata      JSONB DEFAULT '{}',       -- page_start, page_end, section, etc.
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  
  UNIQUE(source_file_id, chunk_index)
);

CREATE INDEX idx_chunks_project ON chunks(project_id);
CREATE INDEX idx_chunks_source ON chunks(source_file_id);
```

### 2.4 `jobs`

Toute tâche longue (Celery). Un projet a N jobs.

```sql
CREATE TABLE jobs (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  type            job_type NOT NULL,
  status          job_status NOT NULL DEFAULT 'queued',
  celery_task_id  VARCHAR(255),           -- ID retourné par Celery
  progress_pct    INTEGER DEFAULT 0,      -- 0-100, mis à jour par le worker
  error_message   TEXT,
  result_summary  JSONB DEFAULT '{}',     -- résumé du résultat (nb exemples, métriques, etc.)
  retry_count     INTEGER DEFAULT 0,
  max_retries     INTEGER DEFAULT 3,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  started_at      TIMESTAMPTZ,
  finished_at     TIMESTAMPTZ,
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_jobs_project ON jobs(project_id);
CREATE INDEX idx_jobs_status ON jobs(project_id, status);
CREATE INDEX idx_jobs_type ON jobs(project_id, type);
```

### 2.5 `dataset_examples`

Chaque exemple du dataset (preview ou final). Stocké en DB pour permettre review/edit/delete.

```sql
CREATE TABLE dataset_examples (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id    UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id        UUID REFERENCES jobs(id) ON DELETE SET NULL,
  source_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
  split         dataset_split NOT NULL DEFAULT 'train',
  data          JSONB NOT NULL,           -- {"instruction":"...", "input":"...", "output":"..."}
  is_valid      BOOLEAN NOT NULL DEFAULT TRUE,
  validation_error TEXT,                  -- raison si is_valid=false
  token_count   INTEGER DEFAULT 0,        -- nb tokens de l'exemple complet
  content_hash  VARCHAR(64),              -- SHA-256 pour dedup
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_examples_project ON dataset_examples(project_id);
CREATE INDEX idx_examples_split ON dataset_examples(project_id, split);
CREATE INDEX idx_examples_valid ON dataset_examples(project_id, is_valid);
CREATE INDEX idx_examples_hash ON dataset_examples(content_hash);
```

### 2.6 `runs`

Chaque exécution de training. Un projet peut avoir plusieurs runs (avec des hyperparams différents).

```sql
CREATE TABLE runs (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id          UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  method          finetune_method NOT NULL,
  status          job_status NOT NULL DEFAULT 'queued',
  hyperparams     JSONB NOT NULL DEFAULT '{}',
  metrics         JSONB DEFAULT '{}',       -- {"eval_loss": 0.45, "train_loss": 0.32, ...}
  hardware_info   JSONB DEFAULT '{}',       -- snapshot du GPU utilisé
  artifacts_dir   VARCHAR(1024),            -- chemin vers les artifacts
  num_train_examples INTEGER DEFAULT 0,
  num_eval_examples  INTEGER DEFAULT 0,
  started_at      TIMESTAMPTZ,
  finished_at     TIMESTAMPTZ,
  duration_seconds INTEGER,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_runs_project ON runs(project_id);
CREATE INDEX idx_runs_job ON runs(job_id);
```

## 3. Diagramme de relations

```
projects 1──────────N files
    │
    ├──1───────────N chunks
    │                  │
    ├──1───────────N jobs
    │                  │
    ├──1───────────N dataset_examples ───── M:1 ──── chunks
    │                  │
    └──1───────────N runs ────────────── 1:1 ──── jobs (type=train)
```

## 4. Triggers et contraintes

```sql
-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_projects_updated
  BEFORE UPDATE ON projects
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_jobs_updated
  BEFORE UPDATE ON jobs
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();
```

## 5. Requêtes fréquentes (référence)

```sql
-- Dashboard: liste projets avec stats
SELECT p.*, 
  COUNT(DISTINCT f.id) FILTER (WHERE f.kind = 'raw_doc') AS doc_count,
  COUNT(DISTINCT de.id) FILTER (WHERE de.is_valid AND de.split = 'train') AS train_examples,
  (SELECT j.status FROM jobs j WHERE j.project_id = p.id ORDER BY j.created_at DESC LIMIT 1) AS last_job_status
FROM projects p
LEFT JOIN files f ON f.project_id = p.id
LEFT JOIN dataset_examples de ON de.project_id = p.id
GROUP BY p.id
ORDER BY p.created_at DESC;

-- Dataset review: exemples paginés avec filtres
SELECT * FROM dataset_examples
WHERE project_id = $1 AND split = 'train' AND is_valid = TRUE
ORDER BY created_at
LIMIT 50 OFFSET 0;

-- Stats dataset
SELECT 
  split,
  COUNT(*) AS total,
  COUNT(*) FILTER (WHERE is_valid) AS valid,
  COUNT(*) FILTER (WHERE NOT is_valid) AS invalid,
  AVG(token_count) AS avg_tokens,
  MIN(token_count) AS min_tokens,
  MAX(token_count) AS max_tokens
FROM dataset_examples
WHERE project_id = $1
GROUP BY split;
```

## 6. Migration Alembic (convention)

```
backend/app/db/migrations/
  versions/
    001_initial_schema.py
    002_add_chunks_table.py
    ...
  env.py
  script.py.mako
alembic.ini
```

- Chaque migration est **idempotente** ou down-migratable
- Naming: `{numero}_{description}.py`
- Toujours tester `upgrade` + `downgrade`

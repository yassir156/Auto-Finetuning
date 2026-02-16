# FineTuneFlow — Formats Dataset & Validation

## 1. Formats supportés (MVP)

### 1.1 Format interne (stocké en DB — `dataset_examples.data` JSONB)

Tous les exemples sont stockés en JSONB dans la table `dataset_examples` avec ce schéma :

```json
{
  "instruction": "string (required, non-empty)",
  "input": "string (optional, peut être vide)",
  "output": "string (required, min 10 chars)"
}
```

### 1.2 Instruction-tuning

Chaque ligne JSONL :
```json
{"instruction": "Explique le concept de gradient descent en machine learning.", "input": "", "output": "Le gradient descent est un algorithme d'optimisation itératif utilisé pour minimiser une fonction de coût..."}
```

### 1.3 Q&A (Question-Answer)

Chaque ligne JSONL :
```json
{"instruction": "Answer the question using the context.", "input": "Context: La photosynthèse est le processus par lequel les plantes convertissent la lumière du soleil en énergie chimique...\nQuestion: Qu'est-ce que la photosynthèse?", "output": "La photosynthèse est le processus par lequel les plantes convertissent la lumière du soleil en énergie chimique, stockée sous forme de glucose."}
```

## 2. Formats d'upload acceptés

### 2.1 JSONL (`.jsonl`)
- Un objet JSON par ligne
- Champs attendus : `instruction`, `input`, `output`
- OU champs alternatifs mappés automatiquement :

| Champs source | → Mapping |
|---------------|-----------|
| `instruction`, `input`, `output` | direct |
| `question`, `answer` | instruction=question, input="", output=answer |
| `prompt`, `completion` | instruction=prompt, input="", output=completion |
| `context`, `question`, `answer` | instruction="Answer using context.", input="Context: {context}\nQuestion: {question}", output=answer |
| `messages` (list of {role, content}) | instruction=user msg, input="", output=assistant msg |

### 2.2 CSV (`.csv`)
- Headers attendus : mêmes champs que JSONL
- Encoding : UTF-8 (détection automatique)
- Séparateur : `,` (détection auto de `;` et `\t`)

### 2.3 JSON (`.json`)
- Array de objets : `[{"instruction": "...", ...}, ...]`

## 3. Format export (training)

Le fichier exporté pour le training est toujours un JSONL :

**`train.jsonl`** et **`eval.jsonl`** :
```jsonl
{"instruction": "...", "input": "...", "output": "..."}
{"instruction": "...", "input": "...", "output": "..."}
```

### Template de formatage pour SFTTrainer

Le formatting function convertit chaque exemple en texte brut pour le modèle :

```python
def format_instruction(example: dict) -> str:
    """Format pour SFTTrainer."""
    if example.get("input"):
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )
```

## 4. Règles de validation

### 4.1 Validation par exemple

| Règle | Condition | Action |
|-------|-----------|--------|
| `instruction` requis | `len(instruction.strip()) > 0` | Rejeter |
| `output` requis | `len(output.strip()) >= 10` | Rejeter |
| `output` max length | `len(output) <= 10000` chars | Rejeter |
| `instruction` max length | `len(instruction) <= 5000` chars | Rejeter |
| `input` max length | `len(input) <= 10000` chars | Rejeter |
| JSON valide | `json.loads(line)` ne lève pas | Rejeter → repair |
| Champs requis présents | `instruction` et `output` in keys | Rejeter |
| Token count | `token_count <= max_seq_length` | Warning + truncate |
| Pas de refus LLM | Pas de "I cannot", "As an AI" | Warning (flag) |
| Pas de hallucination markup | Pas de "```json", "Here is", etc. | Nettoyer |

### 4.2 Validation globale (dataset)

| Règle | Condition | Action |
|-------|-----------|--------|
| Doublons exacts | SHA-256 du contenu | Supprimer duplicata |
| Minimum exemples | `count >= 10` pour training | Bloquer training |
| Split train/eval | `eval_count >= 5` | Avertissement |
| Distribution tokens | Std dev pas trop élevée | Avertissement |

### 4.3 Pipeline de validation (ordre)

```
Ligne brute
  │
  ├─▶ 1. JSON parse
  │     ├─ OK → continue
  │     └─ FAIL → repair programmatique → retry parse → repair LLM → FAIL = rejet
  │
  ├─▶ 2. Champs mapping (si format alternatif)
  │
  ├─▶ 3. Nettoyage texte
  │     - Strip whitespace
  │     - Supprimer markdown wrapper ("```json...```")
  │     - Supprimer préfixes LLM ("Here is the answer:")
  │
  ├─▶ 4. Validation champs (longueur, non-vide)
  │     ├─ OK → continue
  │     └─ FAIL → rejet + message erreur
  │
  ├─▶ 5. Token counting
  │     - Compter tokens avec tokenizer du modèle cible (ou tiktoken comme fallback)
  │     - Si > max_seq_length → truncate output (garder instruction intact)
  │
  ├─▶ 6. Content hash (SHA-256)
  │     - Si hash déjà vu → marquer comme duplicate
  │
  └─▶ 7. Insert DB (dataset_examples)
        - is_valid = True/False
        - validation_error = message si invalid
```

## 5. Statistiques exposées

Après génération, l'API expose :

```json
{
  "total_generated": 2000,
  "valid": 1800,
  "invalid": 200,
  "duplicates_removed": 35,
  "by_split": {
    "train": 1620,
    "eval": 180
  },
  "token_distribution": {
    "min": 45,
    "max": 1024,
    "mean": 312,
    "median": 298,
    "p95": 780
  },
  "validation_errors": {
    "output_too_short": 120,
    "json_parse_failed": 45,
    "duplicate": 35
  }
}
```

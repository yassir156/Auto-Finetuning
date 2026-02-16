# FineTuneFlow — Prompts LLM (Ollama Cloud)

> Tous les prompts utilisés pour interagir avec l'API Ollama Cloud.
> Modèle par défaut : configurable (ex: `llama3.1:70b`)

## 1. System Prompt (commun à tous les appels de génération)

```
You are a data generation engine.
You must output ONLY valid JSON Lines (JSONL): one JSON object per line.
No markdown fences, no explanations, no extra text before or after the JSONL.
Each line must be a complete, valid JSON object.
Respect the schema exactly.
If unsure, output fewer examples but ensure every line is valid JSON.
```

## 2. Instruction-tuning — Génération depuis un chunk

### User prompt template

```
TASK: Create {N} training examples for instruction-tuning from the provided source text.

SCHEMA (each line is a JSON object):
{{"instruction": "a clear user request or question", "input": "additional context if needed, or empty string", "output": "a helpful answer grounded in the source text"}}

RULES:
1. Use ONLY facts present in the source text. No hallucinations.
2. "instruction" should be a natural user request (vary phrasing: explain, describe, list, compare, summarize, etc.).
3. "input" may be an empty string "" if the instruction is self-contained.
4. "output" must be a helpful, accurate answer fully supported by the source.
5. Vary difficulty: some simple factual, some requiring synthesis.
6. Vary style: some formal, some conversational.
7. Each example must be on its own line as a valid JSON object.
8. Output EXACTLY {N} lines. No more, no less.
9. Do NOT wrap output in markdown code fences.

SOURCE TEXT:
<<<
{chunk_text}
>>>

Output {N} JSONL lines now:
```

**Variables :**
- `{N}` : nombre d'exemples à générer (ex: 5)
- `{chunk_text}` : texte du chunk

## 3. Q&A — Génération depuis un chunk

### User prompt template

```
TASK: Generate {N} question-answer pairs grounded in the provided source text.

SCHEMA (each line is a JSON object):
{{"instruction": "Answer the question using the context.", "input": "Context: <relevant excerpt>\nQuestion: <question>", "output": "<answer>"}}

RULES:
1. The answer MUST be fully supported by the context provided in the input.
2. Extract a relevant context excerpt from the source (not the whole source).
3. Questions should vary in type: factual (who/what/when), analytical (why/how), comparative, inferential.
4. Questions should vary in difficulty: easy, medium, hard.
5. Answers should be concise but complete (2-5 sentences typically).
6. Each example must be on its own line as a valid JSON object.
7. Output EXACTLY {N} lines.
8. Do NOT wrap output in markdown code fences.

SOURCE TEXT:
<<<
{chunk_text}
>>>

Output {N} JSONL lines now:
```

## 4. JSON Repair — Prompt LLM (fallback)

Utilisé **uniquement** après échec du repair programmatique.

### System prompt
```
You are a JSON repair tool. You fix broken JSONL text into valid JSONL.
Output ONLY the fixed JSONL lines. No explanations, no markdown.
```

### User prompt template
```
Fix the following broken text into valid JSONL.
Each line must be a valid JSON object matching this schema:

SCHEMA: {{"instruction": "string", "input": "string", "output": "string"}}

RULES:
- Fix JSON syntax errors (missing quotes, brackets, commas).
- If a line is unrecoverable, skip it entirely.
- Do NOT invent new content. Only fix the JSON structure.
- Output ONLY valid JSONL lines.

BROKEN TEXT:
<<<
{bad_text}
>>>

Fixed JSONL:
```

## 5. Paramètres d'appel Ollama Cloud

### Configuration par défaut

```python
OLLAMA_GENERATION_PARAMS = {
    "model": "llama3.1:70b",      # configurable via .env
    "temperature": 0.7,            # diversité modérée
    "top_p": 0.9,
    "max_tokens": 4096,            # assez pour ~5-10 exemples
    "stream": False,               # pas de streaming pour génération batch
}

OLLAMA_REPAIR_PARAMS = {
    "model": "llama3.1:70b",      # même modèle
    "temperature": 0.1,            # très déterministe pour repair
    "top_p": 0.95,
    "max_tokens": 4096,
    "stream": False,
}
```

### Headers

```python
headers = {
    "Authorization": f"Bearer {OLLAMA_CLOUD_API_KEY}",
    "Content-Type": "application/json"
}
```

### Endpoint

```
POST {OLLAMA_CLOUD_BASE_URL}/v1/chat/completions
```

### Request body

```json
{
  "model": "llama3.1:70b",
  "messages": [
    {"role": "system", "content": "<system_prompt>"},
    {"role": "user", "content": "<user_prompt>"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 4096
}
```

### Response parsing

```python
# Extraire le contenu de la réponse
content = response["choices"][0]["message"]["content"]

# Parser les lignes JSONL
examples = []
for line in content.strip().split("\n"):
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        examples.append(obj)
    except json.JSONDecodeError:
        # → repair pipeline
        pass
```

## 6. Stratégie de retry et repair

```
Appel Ollama Cloud
  │
  ├─▶ Timeout (30s) → retry (max 3x, backoff exponentiel)
  │
  ├─▶ Response reçue → parse JSONL
  │     │
  │     ├─▶ Toutes les lignes valides → OK ✓
  │     │
  │     ├─▶ Certaines lignes invalides:
  │     │     1. Repair programmatique:
  │     │        - Strip markdown fences (```json ... ```)
  │     │        - Strip préfixes ("Here are the examples:")
  │     │        - Fix trailing commas
  │     │        - Fix missing closing brackets
  │     │        - Regex extract JSON objects: re.findall(r'\{[^{}]*\}', text)
  │     │     2. Si encore invalide → Appel LLM repair (1 tentative)
  │     │     3. Si encore invalide → rejet de la ligne
  │     │
  │     └─▶ Aucune ligne valide:
  │           1. Retry appel principal (nouveau prompt + hint)
  │           2. Max 2 retries
  │           3. Si échec total → log erreur, passer au chunk suivant
  │
  └─▶ Erreur réseau / 500 → retry (max 3x, backoff)
```

## 7. Optimisation des prompts (tips)

1. **Chunk trop court** (< 100 tokens) : fusionner avec le chunk suivant avant d'envoyer
2. **Chunk trop long** (> 2000 tokens) : limiter N à 3-4 exemples pour ne pas dépasser max_tokens
3. **Diversité** : shuffler l'ordre des chunks, varier N entre 3-7
4. **Coût** : estimer tokens consommés avant de lancer la génération complète
5. **Qualité** : les premiers exemples d'une réponse sont souvent meilleurs → si N trop grand, la qualité baisse

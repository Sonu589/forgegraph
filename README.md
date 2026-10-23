# ForgeGraph

**Evidence-Grounded AI for Air-Gapped Predictive Maintenance** — a Code Vipassana Season 16 Project Pioneer prototype that evolves the 1500T forging-press codelab into a retrieval-augmented maintenance assistant.

> Status: a working local/demo implementation and AlloyDB Omni integration path are included. The GKE/AlloyDB integration requires the infrastructure and model registrations documented in Session 2; it has not been run against your cloud project from this package. Do not describe a cloud deployment as completed until you run and verify it.

## Why it differs from the codelabs

- Session 1: direct prompt-based Gemma analysis of anomalous sensor values.
- Session 2: AlloyDB Omni on GKE, EmbeddingGemma embeddings, and vector similarity search.
- ForgeGraph: combines thresholds, maintenance-document ingestion, vector retrieval, evidence-grounded prompt construction, source excerpts in the UI, explicit empty-model states, and regression tests.

## Features

- Dashboard with pressure, oil-temperature and vibration inputs plus normal/anomaly sample controls.
- Threshold classifier with transparent flags (not a certified failure detector).
- Seed maintenance notes on pressure loss, high oil temperature, vibration spikes, historical incidents and safe escalation.
- Local demo mode: lexical retrieval and a clearly labelled rule-based response; it does **not** pretend to be Gemma.
- AlloyDB mode: writes telemetry to AlloyDB Omni and calls `google_ml.embedding()` for semantic vector search after registering `embeddinggemma` as described by the official codelab.
- Optional Gemma endpoint compatible with OpenAI `/v1/chat/completions` response format.
- A curated symptom/component/incident/procedure knowledge graph traversed to surface possible relationships (explicitly labelled hypotheses, not confirmed causes).
- Document-ingestion API and UI, container image, starting Kubernetes manifest, and GitHub Actions tests.

## Local run (no cloud credits)

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
python app.py
```

Visit `http://127.0.0.1:5000`. Default `APP_MODE=demo` works without a database. Response badges clearly distinguish `RULE-BASED DEMO` from a real `GEMMA RESPONSE`.

Run tests:

```bash
pytest -q
python scripts/check_config.py
```

## Use a real Gemma endpoint

Set `GEMMA_API_URL` to a reachable private endpoint implementing the OpenAI chat-completions shape. For example, with the Session 2 codelab's service forwarded locally, use the corresponding local forwarding URL. Do not expose model-serving endpoints publicly in a regulated deployment.

## AlloyDB Omni / EmbeddingGemma mode

First complete the official setup in [Session 2](https://codelabs.developers.google.com/alloydb-omni-gke-embeddings): deploy AlloyDB Omni on GKE, deploy the EmbeddingGemma endpoint, register the embedding model using `google_ml.create_model`, and deploy/configure Gemma serving. Verify the model's embedding dimension matches `VECTOR(768)` in `sql/schema.sql`.

Configure the database credentials only in environment variables or a Kubernetes secret:

```bash
export APP_MODE=alloydb
export DATABASE_URL='postgresql://postgres:REPLACE_ME@DB_HOST:5432/demo'
export EMBEDDING_MODEL_ID=embeddinggemma
export GEMMA_API_URL='http://gemma-service:8000/v1/chat/completions'
python scripts/seed_database.py
python app.py
```

`DATABASE_URL` is an example, not a real credential. `DB_HOST` and `gemma-service` must match your own private network/service names. In AlloyDB mode, the `/api/analyze` endpoint returns a service error if vector retrieval fails rather than pretending to have performed semantic retrieval.

## Kubernetes starting point

`k8s/forgegraph-app.yaml` shows a non-root app deployment, resource requests/limits, health checks and a Kubernetes secret reference. Before applying it, build and push the Docker image, replace the placeholder image name, create secret `forgegraph-secrets` with key `database-url`, and verify the internal Gemma service name and networking.

```bash
docker build -t YOUR_REGISTRY/forgegraph:latest .
docker push YOUR_REGISTRY/forgegraph:latest
kubectl apply -f k8s/forgegraph-app.yaml
```

This manifest deploys only the Flask app, not AlloyDB Omni or model servers. Use the codelab's deployment guides for those dependencies.

## API

- `GET /api/health` — returns demo/database status and whether a Gemma endpoint is configured.
- `POST /api/analyze` — JSON: `{"pressure": 252, "temperature": 56, "vibration": 5.6}`.
- `POST /api/documents` — JSON: `{"title":"...", "source":"...", "text":"..."}`.
- `POST /api/telemetry/sample` — JSON: `{"preset":"normal"}` or `{"preset":"anomaly"}`.

Sample data is synthetic; the app never controls hardware or creates work orders.

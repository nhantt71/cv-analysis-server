## CV Analysis Server

FastAPI service that scores CVs against jobs, filters candidates, and surfaces related roles by orchestrating PostgreSQL data, Redis caching, and Google Gemini models. Designed to sit behind a recruitment platform and provide AI-assisted decision support.

### Features
- **CV ↔ Job Matching** – persists Gemini-backed NLI scores in PostgreSQL and caches top matches in Redis for fast retrieval.
- **Candidate Filtering** – evaluates CV text against recruiter-provided criteria using bias-aware prompts.
- **Related Role Discovery** – compares job descriptions to surface similar openings.
- **Cache Management API** – inspect, delete, or check TTL for Redis keys without leaving the UI.

### Architecture
- `FastAPI` application (`main.py`) exposes HTTP endpoints.
- `services/database.py` handles PostgreSQL access, Redis caching, and LangChain agent configuration.
- `services/gemini_analysis.py` contains scoring, filtering, and related-job logic powered by `langchain-google-genai`.
- `models/schemas.py` defines response models shared with clients.

### Prerequisites
- Python 3.11+
- PostgreSQL instance with the required tables (`job`, `cv_job_matches`, `nli_analysis`).
- Redis instance for caching.
- Google Cloud project with Gemini access and a service account JSON key.
- (Windows) GTK runtime if you intend to use the optional resume-to-image utilities.

### Environment Variables
Create a `.env` file in the project root with:

```
POSTGRES_HOST=<postgres-host>
POSTGRES_PORT=<postgres-port>
POSTGRES_DB=<postgres-database>
POSTGRES_USER=<postgres-user>
POSTGRES_PASSWORD=<primary-password>
POSTGRES_PASSWORD2=<password-used-by-psycopg2-connect>

REDIS_HOST=<redis-host>
REDIS_PORT=<redis-port>
REDIS_DB=<redis-database-number>

GOOGLE_APPLICATION_CREDENTIALS=<absolute-path-to-service-account.json>
```

> `POSTGRES_PASSWORD` is used by SQLAlchemy; `POSTGRES_PASSWORD2` is used by the direct `psycopg2` connection helpers.

### Setup
```
python -m venv .venv
.venv\Scripts\activate  # PowerShell on Windows; use `source .venv/bin/activate` on POSIX
pip install --upgrade pip
pip install -r requirements.txt
```

### Running Locally
```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The interactive API docs will be available at `http://localhost:8000/docs`.

### API Overview
- `GET /api/recommend/jobs-for-cv/{cv_id}`  
  Returns the top job matches (from cache when available).

- `GET /api/recommend/cvs-for-job/{job_id}`  
  Lists CVs most aligned with a job opening.

- `POST /api/filter`  
  Body: `{ "filters": { ... } }`. Evaluates stored CV texts against recruiter filters and returns the highest scoring candidates.

- `GET /api/related-jobs/{job_id}`  
  Compares a job to other active listings and returns similar roles.

- `POST /api/analyze/cv`  
  Body: `{ "cv_text": "...", "cv_id": 123 }`. Triggers a fresh Gemini analysis across all active jobs, saves results, and caches matches for 30 minutes.

- Cache utilities:  
  - `GET /api/cache/check-exists?key=<redis-key>`  
  - `DELETE /api/cache/delete?key=<redis-key>`  
  - `GET /api/cache/list-keys?pattern=*`  
  - `GET /api/cache/get-ttl?key=<redis-key>`

### Data Flow
1. CV text is analyzed against active jobs fetched from PostgreSQL.  
2. High-confidence matches (score ≥ 0.5) are stored in `cv_job_matches`.  
3. Results and recommendations are cached in Redis for quick follow-up queries.  
4. Subsequent recommendation requests read directly from the cache when present.

### Development Notes
- Ensure the `job`, `cv_job_matches`, and `nli_analysis` tables exist; migrations are not managed in-repo.
- Gemini calls rely on LangChain `ChatGoogleGenerativeAI`. Make sure the service account has the Generative AI API enabled.
- Redis keys are namespaced (`recommend:cv:*`, `related_jobs:*`, etc.) for easy cache invalidation.
- The resume builder utilities in `services/gemini_analysis.py` are currently commented out; uncomment and configure fonts/GTK if you plan to export resumes to PDF/image.

### Testing & Validation
- Use Postman or `curl` to hit endpoints; confirm cache operations using the Redis CLI.
- Consider seeding the DB with sample jobs and CV matches for deterministic tests.
- Add unit tests under a future `tests/` directory to cover database access and service-layer logic; no automated tests ship with this repository yet.

### Deployment Tips
- Run behind a production server such as `uvicorn` with `gunicorn` or `waitress` as listed in `requirements.txt`.
- Configure environment-specific Redis/DB credentials via secrets managers.
- Set `GOOGLE_APPLICATION_CREDENTIALS` to a path accessible on the deployment target and keep credentials out of source control.

### License
No license information is provided. Add one if you plan to share or open-source this project.
# 🔐 Security Guide — HealthSense AI

## ⚠️ CRITICAL: Never Commit Secrets

The following must **NEVER** be committed to Git:

| Type | Example |
|------|---------|
| `.env` file | `SUPABASE_SERVICE_KEY=eyJ...` |
| API keys | Supabase, OpenAI, etc. |
| JWT secrets | Random 32+ char strings |
| Database passwords | Supabase DB password |
| Service account JSON | Firebase / GCP credentials |

All of these are listed in `.gitignore` and will be blocked from commits.

---

## ✅ Safe Setup Steps

### Local Development
```bash
# 1. Copy the template
cp .env.example .env

# 2. Fill in your real values in .env (never commit this file)
# Edit: SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY, JWT_SECRET

# 3. Run backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### On Render (Production)
1. Go to your Render Dashboard → Select your backend service
2. Click **Environment** tab
3. Add each variable from `.env.example` manually
4. **Do NOT** upload `.env` file — type values directly

---

## 🛡 What's Protected

### `.gitignore` covers:
- `.env`, `.env.local`, `.env.production`, `*.env`
- `secrets.json`, `secrets.yaml`, `*-key.json`
- `service-account*.json`, `*credentials*.json`
- ML model files (`.pkl`, `.h5`, `.joblib`)
- Database dumps (`*.sql.gz`, `*.dump`)
- Python cache, venv, build artifacts

### Frontend Security:
- No API keys in any HTML/JS files
- Supabase `anon` key (safe to expose) — all reads/writes protected by **Row Level Security (RLS)**
- Supabase `service_role` key — **backend only**, never in frontend

### Backend Security:
- All secrets loaded via `os.getenv()` — never hardcoded
- Rate limiting: 60 req/min per IP (configurable via `RATE_LIMIT_PER_MINUTE`)
- GZip compression
- CORS restricted to `ALLOWED_ORIGINS`

---

## 🔍 Before Every `git push` — Quick Checklist

```bash
# Check if any .env file is accidentally staged
git status

# Scan for accidentally added secrets
git diff --cached | grep -iE "(api_key|secret|password|token|eyJ)"

# Verify .env is not tracked
git ls-files .env
# Should return nothing
```

---

## 🚨 If a Secret Was Accidentally Pushed

1. **Immediately rotate/revoke** the key at the provider (Supabase, etc.)
2. Remove from Git history:
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch .env" \
     --prune-empty --tag-name-filter cat -- --all
   git push origin --force --all
   ```
3. Generate a new secret and update Render environment variables

---

*This file is safe to commit — it contains no secrets, only instructions.*

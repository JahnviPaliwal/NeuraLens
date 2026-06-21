# Deploying NeuraLens

This project is set up to deploy to **two different platforms** out of the box:

1. **Vercel** — serverless, free, fastest to set up
2. **Hugging Face Spaces** — Docker-based, free, great for showcasing ML projects

Both deployment configs are already included in this folder (`vercel.json`, `Dockerfile`, `.dockerignore`, `README_HF.md`). You don't need to write any new config — just follow the steps below.

---

## Option A — Deploy to Vercel

### Step 1: Push this folder to GitHub
Vercel deploys from a Git repo, not a zip upload.

```bash
cd neuralens-v2
git init
git add .
git commit -m "NeuraLens v2 — initial commit"
```

Create a new empty repo on GitHub (no README/license, just empty), then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/neuralens.git
git branch -M main
git push -u origin main
```

### Step 2: Import the project on Vercel
1. Go to **vercel.com** and sign in (GitHub login is easiest).
2. Click **Add New → Project**.
3. Select your `neuralens` repo from the list and click **Import**.
4. Vercel will auto-detect the `vercel.json` in the repo root — leave all settings as default (Framework Preset: "Other" is fine, it's already configured via `vercel.json`).
5. Click **Deploy**.

### Step 3: Wait for the build
Takes about 30–60 seconds. Vercel will:
- Build `backend/server.js` as a serverless function (via `@vercel/node`)
- Serve `frontend/` as static files (via `@vercel/static`)
- Route `/api/*` to the backend, everything else to the frontend

### Step 4: Open your live URL
Vercel gives you a URL like `https://neuralens-yourname.vercel.app`. Open it — you should see the white/red/blue homepage with the sidebar populated and the green "API connected" dot in the topbar.

### If something looks wrong
- **Blank page / 404** → check the **Deployments → [latest] → Build Logs** tab on Vercel for errors.
- **Sidebar loads but "API connected" dot stays red** → open browser dev tools → Network tab → check if `/api/health` returns 200. If it 404s, double-check `vercel.json` is in the repo root (not inside `backend/` or `frontend/`).
- **Every push to `main` auto-redeploys** — no need to repeat these steps after the first setup.

---

## Option B — Deploy to Hugging Face Spaces

HF Spaces runs this app inside a **Docker container** (config already provided in `Dockerfile`).

### Step 1: Create a new Space
1. Go to **huggingface.co** and sign in (or create a free account).
2. Click your profile icon → **New Space**.
3. Fill in:
   - **Space name**: `neuralens` (or anything you like)
   - **License**: MIT (or your choice)
   - **Select the Space SDK**: choose **Docker**
   - **Docker template**: choose **Blank**
   - **Visibility**: Public (so you can share the link) or Private
4. Click **Create Space**.

### Step 2: Replace the default README with the HF-specific one
HF Spaces reads YAML metadata from `README.md` at the repo root to know how to run it. This project already includes that file as **`README_HF.md`** — you just need to rename it.

```bash
cd neuralens-v2
mv README_HF.md README.md
```

> If you also want to keep the detailed project README, rename the original one to something like `DOCS.md` first, so you don't lose it — the HF one only needs the YAML header + a short description.

### Step 3: Push the project to your new Space
HF Spaces are Git repos too. After creating the Space, HF shows you the clone URL — use it to push:

```bash
git init
git add .
git commit -m "NeuraLens v2 — Docker deploy"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/neuralens
git push -u origin main
```

(If prompted for credentials, use a Hugging Face **access token** as the password — generate one at huggingface.co → Settings → Access Tokens.)

### Step 4: Wait for the build
Go to your Space's page — you'll see a **"Building"** status with live logs. HF will:
- Run your `Dockerfile`
- Install dependencies via `npm install --omit=dev`
- Start the container with `node backend/server.js`
- Expose it on port `7860` (already configured via `ENV PORT=7860` in the Dockerfile)

This takes 1–3 minutes the first time.

### Step 5: Open your live Space
Once the status shows **"Running"**, your app is live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/neuralens
```

### If something looks wrong
- **Build fails** → click the **Logs** tab on your Space to see the exact Docker build error.
- **"Running" but blank/error page** → check that the YAML frontmatter at the top of your `README.md` has `app_port: 7860` exactly matching the Dockerfile's `EXPOSE 7860`.
- **Need to redeploy after a change** → just `git push` again; HF rebuilds automatically.

---

## Quick comparison

| | Vercel | Hugging Face Spaces |
|---|---|---|
| Best for | Fast, simple hosting | Showcasing ML/AI projects, more visibility in ML community |
| Backend model | Serverless functions | Always-on Docker container |
| Cold starts | Possible (serverless) | None once running |
| Setup files used | `vercel.json` | `Dockerfile`, `.dockerignore`, `README.md` (renamed from `README_HF.md`) |
| Custom domain | Yes (free tier supports it) | Yes (paid feature) |

You can deploy to **both** — they don't conflict. Many people put the Vercel link in a resume/portfolio and the HF Space link in their Hugging Face profile.

---

## Local testing before you deploy

Always confirm it runs locally first:

```bash
npm install
npm run dev
```

Visit `http://localhost:3000` — if the sidebar populates and the topbar shows "API connected" in green, you're good to deploy.

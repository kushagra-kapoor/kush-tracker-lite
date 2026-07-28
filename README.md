# ⚡ Kush Tracker Lite (Cloud-Ready CANSLIM Trading Terminal)

`Kush Tracker Lite` is an independent, lightweight, high-performance equity trading application built specifically for deployment on **Streamlit Community Cloud** (or local execution).

---

## 🔑 How to Format Secrets in TOML Format

In Streamlit Cloud (**App Settings ⚙️ -> Secrets**), paste the following TOML configuration:

```toml
[auth]
username = "kush"
password = "your_chosen_password"

[database]
db_type = "turso"
turso_url = "libsql://your-database-name-kush410.turso.io"
turso_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### Where to get your 2 Turso credentials:
1. **`turso_url` (Database URL):**
   * Go to [app.turso.tech](https://app.turso.tech/), click your Database (e.g. `kush-tracker-lite`).
   * Copy the **Database URL** starting with `libsql://` (e.g., `libsql://kush-tracker-lite-kush410.turso.io`).
2. **`turso_token` (Auth / API Token):**
   * Go to `https://app.turso.tech/kush410/settings/api-tokens` (or create a DB token in Turso CLI).
   * Copy the token string (starts with `eyJ...`).

---

## 🚀 Deployment Instructions for Streamlit Community Cloud

### Step 1: Push Code to GitHub
Run `push_to_github.bat` or execute:
```bash
git push -u origin main
```

### Step 2: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Click **"New App"**.
3. Select your repository: `kush410/kush-tracker-lite`.
4. Main file path: `app.py`.
5. Add your `[auth]` & `[database]` secrets in TOML format above.
6. Click **"Deploy!"**.

---

## 💻 Running Locally

Double-click `run_kush_tracker_lite.bat` or run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

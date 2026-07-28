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
turso_url = "libsql://kush-db-kush410.aws-ap-northeast-1.turso.io"
turso_token = "PASTE_TOKEN_FROM_CONNECT_TAB_HERE"
```

### Which Turso Token to use?
* Use the **Database Auth Token** generated under **Connect -> Create Token** (right next to your Database URL).
* *Do NOT use the account platform API token under Settings -> API Tokens.*

---

## 🚀 Deployment Instructions for Streamlit Community Cloud

### Step 1: Push Code to GitHub
Double-click `push_to_github.bat` or run:
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

# ⚡ Kush Tracker Lite (Cloud-Ready CANSLIM Trading Terminal)

`Kush Tracker Lite` is an independent, lightweight, high-performance equity trading application built specifically for deployment on **Streamlit Community Cloud** (or local execution).

It implements the full **CANSLIM & Minervini Trading Framework** across 6 specialized modules for **India (NSE)** and **US (NYSE/NASDAQ)** equities:

1. **🛡️ Market Regime & Health Gauge:** CANSLIM 'M' Market Direction HUD, FOMO/FEAR Gauge, Deep RS Leaders Cards (IN & US), Total Market Breadth, and 120-Day Market Extremes.
2. **⚡ Intraday Monitor (India):** Live market breadth, volume surges, gainers/losers, distribution day counter, and **automated volume shock breakout logging**.
3. **⚡ Intraday Monitor (US):** Live US market breadth, US volume surge scanner, and **automated volume shock breakout logging**.
4. **👑 True Market Leaders (India):** CANSLIM 'C-A-L-I' institutional leadership scanner with Clenow Momentum Exponential Slopes ($R^2$), percentile RS scores, and Stage Analysis.
5. **👑 True Market Leaders (US):** US institutional momentum scanner for S&P 500 & NASDAQ leaders.
6. **⭐ Focus List & Breakout Execution Hub:** Dedicated execution workspace displaying **auto-populated volume shock breakouts** + user-pinned setups with entry triggers, stop losses, and live 15s price alerts.

---

## 🔒 Authentication & Login Setup (Streamlit Secrets)

Since this app is deployed publicly on Streamlit Cloud, a **Glassmorphic Login Gate** is built into `app.py`.

Set your custom private **Username** and **Password** in Streamlit Cloud Settings (**App Settings ⚙️ -> Secrets**):

```toml
[auth]
username = "your_custom_username"
password = "your_secure_password"

[database]
db_type = "turso"
turso_url = "libsql://kush-tracker-lite-YOURNAME.turso.io"
turso_token = "your-turso-jwt-token"
```

*(Default fallback for local testing: `admin` / `admin123`)*

---

## 🚀 Deployment Instructions for Streamlit Community Cloud

### Step 1: Push Code to GitHub
Create a new repository on GitHub (e.g. `kush-tracker-lite`) and push this directory:
```bash
git init
git add .
git commit -m "Initial commit of Kush Tracker Lite"
git remote add origin https://github.com/YOUR_USERNAME/kush-tracker-lite.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Click **"New App"**.
3. Select your repository: `YOUR_USERNAME/kush-tracker-lite`.
4. Main file path: `app.py`.
5. Add your `[auth]` & `[database]` secrets.
6. Click **"Deploy!"**.

---

## 💻 Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

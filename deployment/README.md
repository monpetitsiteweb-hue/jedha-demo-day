# Deployment

Two independent pieces run in production:

1. **The ML inference API** — FastAPI on **Railway**, called by the trading engine. (See [`../api/`](../api/) for the full service doc.)
2. **The analytics dashboard** — this project's **Streamlit** app, on a self-managed **VPS** behind nginx with HTTPS → https://jedha.carlos-isturiz.fr

```
Trading engine ──▶ ML API (Railway, FastAPI)          Dashboard (VPS, Streamlit + nginx + HTTPS)
                   XGBoost + LSTM ensemble             reads the same models/data, runs the ensemble live
```

---

## 1. ML inference API — Railway

- **Build:** Docker (`railway.toml` → `builder = "dockerfile"`), started with `CMD ["python", "main.py"]` → Uvicorn on `0.0.0.0:$PORT`.
- **Environment variables:** `SUPABASE_URL`, `SUPABASE_KEY` (service_role, secret), optional `MODELS_DIR`, and `PORT` (injected by Railway). See `api/.env.example`.
- **Models:** downloaded at start-up from the Supabase Storage bucket `ml-models` (6 files). No hot-reload — to update a model: upload to the bucket, then restart the Railway service.
- **Endpoint:** `POST /predict` (public, unauthenticated — a known limitation), `GET /healthz` for health.

Full details, request/response examples, and the security section are in [`../api/README.md`](../api/README.md).

---

## 2. Analytics dashboard — VPS (Streamlit + nginx + HTTPS)

**Host:** Ubuntu 24.04 VPS. App lives in `/opt/jedha-app/` with a Python virtualenv.

### One-time setup

**a) App + virtualenv**
```bash
cd /opt/jedha-app
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# pinned to match the training environment (scikit-learn, xgboost, numpy, pandas, streamlit)
```

**b) Data & models** — pushed from the workstation with `scp` into:
```
/opt/jedha-app/data/processed/   (test_features_*.csv, feature_cols_selected.json, latest.json, …)
/opt/jedha-app/models/           (xgb_*.pkl, rf_*.pkl, lstm_final*.keras, scaler_lstm*.pkl, models_latest.json)
```

**c) TensorFlow (for the live ensemble)** — the dashboard runs the XGBoost + LSTM ensemble live, so it needs TensorFlow:
```bash
pip install tensorflow-cpu==2.21.0
python -c "from tensorflow import keras; keras.models.load_model('models/lstm_final.keras'); print('LSTM OK')"
```
> Note: TensorFlow pulls a newer `protobuf` than Streamlit pins; in practice Streamlit still runs. If it doesn't, use `tensorflow-cpu==2.18.*` (keeps `protobuf<6`).

**d) systemd service** — `/etc/systemd/system/jedha.service` runs:
```
streamlit run app_streamlit.py --server.address 0.0.0.0 --server.port 8501 \
  --server.headless true --server.enableCORS false --server.enableXsrfProtection false
```
with `Restart=always`. Then:
```bash
systemctl daemon-reload && systemctl enable --now jedha
```

**e) nginx reverse proxy** — a site config proxies `:443` → `localhost:8501`, forwarding WebSocket upgrade headers (Streamlit needs them). Enable the site and reload nginx.

**f) HTTPS** — Let's Encrypt via certbot (`certbot --nginx`), auto-renew enabled.

**g) Firewall** — `ufw` allows 22 / 80 / 443. DNS: an `A` record points the domain to the VPS IP.

### Redeploy (routine)

From the **workstation** (PowerShell):
```powershell
scp "path\to\app_streamlit.py" root@<VPS_IP>:/opt/jedha-app/app_streamlit.py
```
Then on the **VPS**:
```bash
systemctl restart jedha
```
Hard-refresh the browser (Ctrl+Shift+R).

### Updating the models on the dashboard

Push the new artifacts and point `models_latest.json` / `data/processed/latest.json` at them, then restart:
```powershell
scp models\xgb_final_YYYYMMDD.pkl      root@<VPS_IP>:/opt/jedha-app/models/
scp models\lstm_final_YYYYMMDD.keras   root@<VPS_IP>:/opt/jedha-app/models/
scp models\models_latest.json          root@<VPS_IP>:/opt/jedha-app/models/
# + the matching test_features_*.csv and feature_cols_selected.json into data/processed/
```
```bash
systemctl restart jedha
```
The app auto-detects local vs VPS paths, so the same `app_streamlit.py` runs in both environments.

### Troubleshooting

- `systemctl status jedha --no-pager` — is the service running?
- Dashboard shows "LSTM not available" → TensorFlow missing, or the model/feature counts don't match (the app falls back to the tree models and keeps running).
- `free -h` while the "earn" page loads — TensorFlow is memory-hungry; make sure the VPS has headroom.

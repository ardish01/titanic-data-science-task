# Practical Execution Task — Data Science (JobAxle)

End-to-end Titanic workflow: EDA → missing values & duplicates → encodings → plots → outliers → feature engineering → scaling → train/test split → logistic regression & decision tree → K-Means + elbow → **FastAPI** (task 13) → **Requests + BeautifulSoup** scrape (task 14) → **Selenium** (task 15).

| Location | What it covers |
|----------|----------------|
| `titanic_data_science_tasks.ipynb` | Tasks **1–12** and **14** (notebook), including saving `model_artifacts.joblib` after training when you run all cells |
| `build_artifacts.py` | Same training pipeline as the notebook; writes **`model_artifacts.joblib`** for `app.py` (use after clone or without opening Jupyter) |
| `app.py` | Task **13**: FastAPI, **real** sklearn predictions |
| `selenium_task.py` | Task **15**: login + data extraction on a public demo site |

## How to run (quick start)

All commands assume PowerShell, project folder `titanic-data-science-task`, and venv already created (see **Setup** below).

| What | Commands |
|------|----------|
| **One-time install** | `python -m venv .venv` → `.\.venv\Scripts\Activate.ps1` → `pip install -r requirements.txt` |
| **Notebook (tasks 1–12, 14)** | `.\.venv\Scripts\Activate.ps1` → `jupyter notebook titanic_data_science_tasks.ipynb` → in the menu: **Kernel → Restart & Run All** |
| **API (task 13)** | `.\.venv\Scripts\Activate.ps1` → `python build_artifacts.py` → **`python -m uvicorn app:app --reload`** *or* **`python app.py`** → open **http://127.0.0.1:8000/docs** |
| **Selenium (task 15)** | `.\.venv\Scripts\Activate.ps1` → `python selenium_task.py` |

Use **separate terminals** for Jupyter and for Uvicorn if both run at once. Stop the API with **Ctrl+C**.

## Prerequisites

- Python **3.10+** recommended  
- **Task 15 (Selenium):** this project uses **Microsoft Edge only**. Edge is included with Windows. **Do not use Google Chrome or Brave** for this script—`selenium_task.py` is written for Edge via Selenium’s Edge driver.

## Setup (Windows PowerShell)

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

You can name the env `venv` instead of `.venv`; activate with `.\venv\Scripts\Activate.ps1` in that case.

## Notebook

1. Activate the venv (see above).  
2. Start Jupyter, e.g. `jupyter notebook titanic_data_science_tasks.ipynb`.  
3. **Kernel → Restart & Run All.**

The notebook imputes **Age**, **Fare**, and **Embarked**, drops high-**Fare** outliers for later steps, then trains models. Task **13** in the notebook only *points* to `app.py`; the runnable API is the script in this repo.

### Jupyter: `Permission denied` on `%APPDATA%\jupyter\runtime` (Windows)

In the **same** PowerShell session before `jupyter notebook`:

```powershell
New-Item -ItemType Directory -Force -Path ".\.jupyter_runtime" | Out-Null
$env:JUPYTER_RUNTIME_DIR = "$PWD\.jupyter_runtime"
jupyter notebook titanic_data_science_tasks.ipynb
```

## Task 13 — FastAPI

`model_artifacts.joblib` is **gitignored** (you generate it locally). Create it, then start the server.

```powershell
.\.venv\Scripts\Activate.ps1
python build_artifacts.py
python -m uvicorn app:app --reload
```

Or start the same app without typing `uvicorn` on PATH:

```powershell
python app.py
```

Use **`python -m uvicorn`** (or **`python app.py`**) so you do not rely on the `uvicorn` executable being on `PATH`. If **port 8000 is already in use**, set another port: `$env:PORT="8001"; python app.py`, then open **http://127.0.0.1:8001/docs**.

With the server running, open **http://127.0.0.1:8000/docs** (or the port you chose).

| Method | Path | Purpose |
|--------|------|--------|
| **GET** | `/` | Health check. Example response: `{"message":"Titanic Survival Prediction API - POST /predict with passenger JSON"}` |
| **POST** | `/predict` | Send passenger JSON and get a survival prediction |

**POST `/predict`** body example:

```json
{"pclass": 1, "age": 30, "fare": 50, "sibsp": 0, "parch": 0, "sex": "female"}
```

`sex` must be `"male"` or `"female"` (same labels as the Titanic CSV). The response includes `survived`, `label`, and `survival_probability`.

## Task 15 — Selenium (Microsoft Edge only)

`selenium_task.py` automates **Microsoft Edge** in headless mode (Selenium 4 resolves **msedgedriver**). It does **not** launch Chrome, Brave, or any other browser.

Demo login (credentials published by the site): [the-internet.herokuapp.com/login](https://the-internet.herokuapp.com/login).

```powershell
.\.venv\Scripts\Activate.ps1
python selenium_task.py
```

Ensure **Edge is installed** (default on Windows 11; install or update Edge if the script fails to start the driver). You may see harmless log lines in the terminal; success looks like the login banner text followed by **Browser closed.**

## Submission (JobAxle)

1. Push this project to a **public GitHub** repository (include the notebook, `app.py`, `build_artifacts.py`, `selenium_task.py`, `requirements.txt`, and this `README.md`).  
2. After cloning, reviewers should run **`pip install -r requirements.txt`** and **`python build_artifacts.py`** before **`python -m uvicorn app:app --reload`**.  
3. Share the repository link as instructed by the assessor.

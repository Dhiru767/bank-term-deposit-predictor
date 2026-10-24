# Term Deposit Predictor

A Django + machine learning web application that estimates how likely a bank client is to subscribe to a term deposit.

## Description

A registered user enters details about a bank client (age, job, loans, contact history and a few more fields). A trained Random Forest model returns a probability and a "likely / unlikely to subscribe" result. Every prediction is saved to the user's history in the database.

The model is trained once, offline, by `ml/train_model.py` and saved with joblib. Django only loads the saved file; it never trains anything.

## Features

- Register, log in and log out (Django's built-in authentication)
- Login-protected dashboard with a welcome message, project description, statistics and recent predictions
- Prediction form with validation for all 11 inputs
- Result page with the probability and a plain-language result
- Prediction history for each user, with pagination
- About page that shows the real test results saved by the training script
- Friendly error messages for invalid input, a missing model file and database errors
- 56 automated tests

## Technology Stack

- Python 3.11 or newer (tested on 3.13)
- Django 5.1.4
- SQLite (development database)
- HTML5, CSS3 and Bootstrap 5 (loaded from a CDN, so an internet connection is needed for styling)
- pandas, NumPy, scikit-learn, joblib
- python-dotenv (reads the `.env` file)

## Machine Learning Approach

| Item | Choice |
|---|---|
| Problem type | Binary classification |
| Target | `y`: did the client subscribe to a term deposit (yes / no) |
| Algorithm | Random Forest (300 trees, `min_samples_leaf=5`, `class_weight="balanced"`) |
| Numeric features | age, campaign, previous |
| Categorical features | job, marital, education, housing, loan, contact, month, poutcome |
| Encoding | One-hot encoding of categorical features |
| Scaling | Not needed for tree-based models |
| Missing values | Imputed inside the pipeline (median / most frequent); "unknown" is kept as its own category |
| Split | Stratified 80% train / 20% test, `random_state=42` |
| Saved file | One scikit-learn Pipeline (preprocessing + model) in `ml/model/model.joblib` |

Why Random Forest: it handles a mix of numeric and categorical data, needs no scaling, copes with non-linear patterns and is easy to explain (many decision trees vote).

Design decisions:

- Only about 11% of clients subscribed, so the classes are weighted and the split is stratified. Accuracy alone is misleading here, so precision, recall, F1 and ROC-AUC are reported.
- The `duration` column (call length) is not used, because it is only known after the call has ended and would leak the answer.
- `pdays`, `day_of_week`, `default` and the economic indicator columns are left out so the web form only asks for things a user can reasonably know.
- Rows that are identical in every original column are removed before splitting.

### Results (one run of `ml/train_model.py`, 8,236 test clients)

| Metric | Value |
|---|---|
| Accuracy | 0.854 |
| Precision (subscribers) | 0.394 |
| Recall (subscribers) | 0.548 |
| F1 score | 0.459 |
| ROC-AUC | 0.774 |

Confusion matrix (rows = actual no/yes, columns = predicted no/yes): `[[6525, 783], [419, 509]]`.

The same figures are written to `ml/model/metrics.json` after every training run and are shown on the About page. If you retrain, use the numbers from your own run.

## Dataset

UCI Bank Marketing, file `bank-additional-full.csv` (included in `data/`).

- Source: phone marketing campaigns of a Portuguese bank, May 2008 to November 2010
- Records: 41,188 rows, 21 columns (20 inputs and the target `y`); 41,176 remain after removing 12 duplicate rows
- Separator: semicolon (`;`)
- Classes: about 11% "yes", 89% "no"
- Months: the data contains calls from March to December only, so the form offers only those months

Citation: S. Moro, P. Cortez and P. Rita. "A Data-Driven Approach to Predict the Success of Bank Telemarketing." Decision Support Systems, 2014.

## Project Structure

    term_deposit_predictor/
        manage.py
        requirements.txt
        README.md
        .env.example
        data/
            bank-additional-full.csv
        ml/
            config.py            features, choices and file paths (single source of truth)
            train_model.py       trains and saves the model
            model/
                model.joblib     trained pipeline
                metrics.json     test results
        accounts/                register, login, logout
            forms.py  views.py  urls.py  tests.py
        predictor/
            models.py            PredictionHistory
            forms.py             prediction form
            views.py  urls.py  admin.py
            ml_service.py        loads the model and makes predictions
            test_ml.py  test_views.py
            migrations/
        config/                  Django project settings and root URLs
        templates/               base, home, about, dashboard, predict, result, history
            accounts/            login, register
        static/
        notebooks/               training notebook

## Installation

You need Python 3.11 or newer. Run all commands from the project root (the folder that contains `manage.py`).

### Virtual Environment Setup

Linux / macOS:

    python3.11 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Windows (PowerShell):

    py -3.13 -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

Conda:

    conda create -n project-env python=3.11
    conda activate project-env
    pip install -r requirements.txt

### Environment Variables

The secret key is not stored in the code. Create a `.env` file with a random key (works on every system):

    python -c "import secrets; open('.env','w').write('DJANGO_SECRET_KEY=' + secrets.token_urlsafe(50) + '\nDJANGO_DEBUG=True\nDJANGO_ALLOWED_HOSTS=127.0.0.1,localhost\n')"

Never share or commit `.env`. `.env.example` shows the format.

### Database Setup

    python manage.py migrate

### Train the Model

A trained model is already included in `ml/model/`. To train it again (takes from several seconds to a couple of minutes):

    python ml/train_model.py

Restart the server after retraining, because the model is loaded once.

### Run the Application

    python manage.py runserver

Then open http://127.0.0.1:8000/ in your browser. Press Ctrl+C in the terminal to stop the server.

Using the app:

1. Click Register and create an account.
2. On the dashboard, click New prediction.
3. Fill in the client details and press Predict.
4. Read the result, then open History to see all your past predictions.

### Run the Tests

    python manage.py test

All 56 tests should pass. They use a temporary database, so your real data is not touched.

### Admin Site (optional)

    python manage.py createsuperuser

Then open http://127.0.0.1:8000/admin/ to browse users and saved predictions.

## Troubleshooting

- "DJANGO_SECRET_KEY is missing": create the `.env` file (see Environment Variables).
- "The trained model file was not found": run `python ml/train_model.py`.
- "ModuleNotFoundError": activate the virtual environment and run `pip install -r requirements.txt`.
- "No module named 'ml'": run commands from the project root, not from inside a sub-folder.
- Pages look unstyled: Bootstrap is loaded from the internet; check your connection.
- A scikit-learn version warning when loading the model: retrain with `python ml/train_model.py` using your installed version.

## Limitations

- ROC-AUC is moderate (0.774): the model finds many, but not all, subscribers and also produces false alarms.
- The data comes from one Portuguese bank between 2008 and 2010 and may not match other markets or later years.
- This is a learning project. Do not use it to make real decisions about people.


## Prediction API (FastAPI)

The app calls a separate FastAPI service to get predictions. Start it in a second terminal, from the project root:

    uvicorn api_service.app:app --port 8001

Then start Django as usual (python manage.py runserver). Interactive API docs: http://127.0.0.1:8001/docs
Endpoints: GET /health and POST /predict. If the API is not running, the web app falls back to the local model.

# Repository Guidelines

## Project Structure & Module Organization

- `src/` contains the reusable pipeline: transaction cleaning and customer features (`data.py`), clustering and metrics (`model.py`), MLflow setup (`tracking.py`), training (`train.py`), and inference (`inference.py`).
- `app.py` is the Streamlit dashboard. It reads `outputs/customer_clusters.csv` by default and also accepts an uploaded predictions CSV.
- `data/` holds the source workbook; `eda/` contains exploratory analysis; `experiments/` contains model-development notebooks.
- `outputs/`, `mlflow.db`, and MLflow artifact directories are generated locally and should not be committed.

## Build, Test, and Development Commands

Run these from the repository root after installing `requirements.txt`:

```bash
python -m src.train       # Train, track, and register the K-Means pipeline
python -m src.inference   # Load the registered model and write customer predictions
streamlit run app.py      # Start the results dashboard
python -m compileall src app.py  # Check Python syntax
```

Training and inference use the project-local MLflow SQLite store by default. Run training before inference when no registered model alias exists.

## Coding Style & Naming Conventions

Use four spaces for indentation, `snake_case` for functions and variables, and descriptive names for features and metrics. Keep code, comments, and user-facing dashboard text in English. Add Google-style docstrings to important public functions. Keep preprocessing consistent by applying the registered pipeline in inference; do not duplicate its fitted transformations elsewhere.

## Testing Guidelines

Use the standard-library `unittest` framework for automated tests in `tests/`, named `test_*.py`. Run them with `python -m unittest discover -s tests`; there is no coverage threshold. For changes to data or model behavior, validate the relevant command end to end and check the produced CSV or MLflow run. For dashboard changes, launch the app and confirm that the default and uploaded CSV paths render.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects such as `Add Streamlit dashboard with clustering results` and `Fix typos`; follow that pattern. Pull requests should explain the change, list validation commands and results, link related issues when applicable, and include a dashboard screenshot for visible UI changes.

## Data & Configuration

Do not commit customer-level exports, database files, credentials, or generated model artifacts. Keep paths configurable and avoid embedding private tracking-server credentials in source files. For local Anthropic credentials, use `.streamlit/secrets.toml` with an `[anthropic]` section containing `api_key` and `model`; this file is ignored by Git. Configure the same values in Streamlit Community Cloud's Secrets settings for deployment.

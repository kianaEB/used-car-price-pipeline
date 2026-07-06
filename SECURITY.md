# Security

This project keeps secrets out of the codebase -- a deliberate fix for the original prototype,
which hardcoded a database password in source.

## How secrets are handled

- **No credentials in code or config.** The database URL is built in `src/config.py` from
  environment variables only (`src/config.py::_build_db_url`).
- **`.env` is git-ignored.** Copy `.env.example` (names only, no values) to `.env` and fill it
  locally; `.env` is never committed.
- **Docker Compose** reads the same variables from `.env`; the `cars` fallbacks are dev-only
  defaults for local Postgres, not real secrets.
- **CI runs offline** on synthetic data and needs no credentials at all.

## Rotating a credential

Git history preserves old values, so if a real secret is ever committed, rotate it at the source
(the database) -- removing it from the latest commit is not enough. For a local throwaway dev
credential (e.g. a local Postgres you control), deleting the container/volume is sufficient.

## Note on the original prototype

`reference/cars_original.py` is kept only for the before/after story, with its password REDACTED.
The original public repo it came from (`Cars_Price_prediction_by_ML`) is being retired in favor of
this one; making it private or deleting it removes the old value from public view. If that value
was a local-only throwaway (it was), there is nothing further to rotate.

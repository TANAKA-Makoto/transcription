poetry run black src
poetry run isort src
poetry run flake8 src
poetry run mypy src --ignore-missing-imports

poetry run python src/app.py

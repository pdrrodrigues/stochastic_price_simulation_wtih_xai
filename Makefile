setup:
	docker compose build

up:
	docker compose up -d

shell:
	docker compose exec dev bash

test:
	docker compose exec dev pytest -q --disable-warnings

lint:
	docker compose exec dev ruff check src tests

format:
	docker compose exec dev black src tests

precommit:
	docker compose exec dev pre-commit run -a

docs:
	docker compose exec dev sphinx-build -b html docs docs/_build

pipeline:
	docker compose exec dev python -m scripts.pipeline --config configs/m1.yaml

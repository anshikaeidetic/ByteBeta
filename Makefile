.PHONY: install-dev refresh-locks check-locks hygiene lint format typecheck compile test coverage integration-smoke security package-check ci precommit

ifeq ($(OS),Windows_NT)
BOOTSTRAP_CMD=powershell -ExecutionPolicy Bypass -File .\\bootstrap-dev.ps1
VENV_PYTHON=.venv/Scripts/python.exe
else
BOOTSTRAP_CMD=./bootstrap-dev.sh
VENV_PYTHON=.venv/bin/python
endif

install-dev:
	$(BOOTSTRAP_CMD)

refresh-locks:
	$(VENV_PYTHON) scripts/refresh_locks.py

check-locks:
	$(VENV_PYTHON) scripts/check_locks.py

hygiene:
	$(VENV_PYTHON) scripts/check_repo_hygiene.py
	$(VENV_PYTHON) scripts/check_locks.py

lint:
	$(VENV_PYTHON) scripts/run_lint.py

format:
	$(VENV_PYTHON) -m black .

typecheck:
	$(VENV_PYTHON) scripts/run_typecheck.py

compile:
	$(VENV_PYTHON) scripts/compile_check.py

test:
	$(VENV_PYTHON) scripts/run_unit_tests.py

coverage:
	$(VENV_PYTHON) scripts/run_coverage.py

integration-smoke:
	$(VENV_PYTHON) scripts/run_integration_smoke.py

security:
	$(VENV_PYTHON) scripts/run_security_checks.py

package-check:
	$(VENV_PYTHON) -m build --no-isolation --sdist --wheel
	$(VENV_PYTHON) -m twine check dist/*

ci: hygiene lint typecheck package-check compile test coverage integration-smoke security

precommit:
	$(VENV_PYTHON) -m pre_commit run --all-files

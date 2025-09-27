.PHONY: test quiet clean lint

# Default test target (verbose)
test:
	python3 -m pytest -v -s

# Quiet test run
quiet:
	python3 -m pytest

# Clean pytest cache and Python bytecode
clean:
	rm -rf .pytest_cache
	rm -rf __pycache__
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Check Python syntax
lint:
	python3 -m py_compile test_*.py conftest.py
test:
	pytest tests/*

test-fast:
	pytest -m "not slow" tests/*
test:
	CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest tests/*

test-data-dir:
	ASSERT_DATADIR=1 pytest tests/datasets_datadir.py

test-fast:
	pytest -m "not slow" tests/*


sync-artifact:
	rsync --update -rv --max-size=2m ml-slurm-server:~/projects/xai-kd/artifacts/$(name) ./artifacts


jupyter:
	PYTHONPATH=. poetry run jupyter notebook ./notebooks
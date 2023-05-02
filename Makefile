test:
	pytest tests/*

test-fast:
	pytest -m "not slow" tests/*


sync-artifact:
	rsync --update -rv --max-size=1m ml-slurm-server:~/projects/xai-kd/artifacts/$(name) ./artifacts

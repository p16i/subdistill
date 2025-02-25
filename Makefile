test:
	CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest tests/*

test-data-dir:
	ASSERT_DATADIR=1 pytest tests/datasets_datadir.py

test-fast:
	pytest -m "not slow" tests/*


sync-artifact:
	rsync --update -rv --max-size=2m hydra:~/projects/xai-kd/artifacts/$(name) ./artifacts


jupyter-local:
	DATASET_ROOT=$(shell pwd)/datasets PYTHONPATH=$(shell pwd) jupyter notebook --ip=0.0.0.0 ./notebooks

jupyter-app:
	APPTAINERENV_PYTHONPATH=$(shell pwd) DATASET_ROOT=$(shell pwd)/datasets  ./runpy  jupyter notebook --ip 0.0.0.0   ./notebooks

srun5h:
	srun -p gpu-5h --pty --gres=gpu:1 --constraint="40gb" /bin/bash

srun2d:
	srun -p gpu-2d --pty --gres=gpu:1 --mem=128GB --cpus-per-task=16 --constraint="40gb" /bin/bash



remove-branches:
	git branch --merged=main | grep -v main  | xargs -I{} git branch -d {}
ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
APP_NAME:=gpt4all_api
PYTHON:=python3.11
SHELL := /bin/bash

.PHONY: all fresh testenv testenv_gpu testenv_d test test_build clean_testenv fresh_testenv venv dependencies clean black isort

all: dependencies

fresh: clean dependencies

delete_images:
	@if [ -n "$(shell docker images -q $(APP_NAME))" ]; then \
		docker rmi $(APP_NAME); \
	fi
	@if [ -n "$(shell docker images -q gpt4all_gpu)" ]; then \
		docker rmi gpt4all_gpu; \
	fi

testenv: clean_testenv test_build
	docker-compose -f docker-compose.yaml up --build

testenv_gpu: clean_testenv test_build
	docker-compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build

testenv_d: clean_testenv test_build
	docker-compose up --build -d

test:
	docker-compose exec $(APP_NAME) pytest -svv --disable-warnings -p no:cacheprovider /app/tests

test_build:
	DOCKER_BUILDKIT=1 docker build -t $(APP_NAME) --progress plain -f $(APP_NAME)/Dockerfile.buildkit .

clean_testenv:
	docker-compose down -v

clean_testenv_gpu:
	docker-compose -f docker-compose.yaml -f docker-compose.gpu.yaml down -v

fresh_testenv: clean_testenv testenv

venv:
	test -d $(ROOT_DIR)/venv || $(PYTHON) -m venv $(ROOT_DIR)/venv

dependencies: venv
	. $(ROOT_DIR)/venv/bin/activate; $(PYTHON) -m pip install -r $(ROOT_DIR)/$(APP_NAME)/requirements.txt

clean: clean_testenv delete_images clean_testenv_gpu
	# Remove existing environment
	sudo find . -type d -name "__pycache__" -exec rm -rf {} +
	sudo rm -rf $(ROOT_DIR)/venv
	sudo find $(ROOT_DIR)/$(APP_NAME) -name '*.pyc' -delete


black:
	. $(ROOT_DIR)/venv/bin/activate; black -l 120 -S --target-version py38 $(APP_NAME)

isort:
	. $(ROOT_DIR)/venv/bin/activate; isort --ignore-whitespace --atomic -w 120 $(APP_NAME)

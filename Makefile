all: run

init:
	pipenv sync

run:
	pipenv run python main.py

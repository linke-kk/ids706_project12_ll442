# Setup virtual environment
setup:
	python3 -m venv env
	# source env/bin/activate

# install all the packages
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	# ruff check --fix .

format:	
	./env/bin/black *.py 

test:
	# ./env/bin/python -m pytest test_main.py 
	
		
all: setup install lint format test 

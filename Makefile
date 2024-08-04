.PHONY: help install editable requirements test clean triple-extraction context-generation inference-zero-shot inference-fine-tuning format lint

help:  ## Display this help.
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: editable requirements ## Install the package in editable mode and install dependencies from requirements.txt

editable:  ## Install the package in editable mode
	pip install -e .

requirements:  ## Install dependencies from requirements.txt
	pip install -r requirements.txt

test:  ## Run all tests
	pytest tests

test_data_extraction:  
	pytest tests/test_data_extraction.py

test_evaluation:  ## Run evaluation tests
	pytest tests/test_evaluation.py

test_evidence_matching:  
	pytest tests/test_evidence_matching.py

test_fine_tuning:  
	jupyter nbconvert --to notebook --execute tests/test_fine_tuning_T5.ipynb

test_predictions:  
	pytest tests/test_predictions.py

test_prepare_context:  
	pytest tests/test_prepare_context.py

test_qa_pipeline:  
	pytest tests/test_qa_pipeline.py

test_zero_shot_prompting:  
	pytest tests/test_zero_shot_prompting.py

clean:  
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

triple-extraction:  

context-generation: 


inference-zero-shot:  


inference-fine-tuning:  


format: 
	black .
	isort .

lint:  
	flake8 .
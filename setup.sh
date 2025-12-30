#!/bin/bash
# Setup script.
# Creates virtual environment, installs dependencies, and downloads spacy model

set -e

echo "Creating directories"
mkdir -p reddit_data
mkdir -p fitted_models
mkdir -p embeddings
mkdir -p evaluations
mkdir -p figures
mkdir -p topic_csvs

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment"
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

#install dependecencies
echo "installing dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "Installing Jupyter kernel"
pip install ipykernel
python -m ipykernel install --user --name=nlp_project --display-name="NLP Project"


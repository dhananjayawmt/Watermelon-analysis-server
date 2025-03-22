#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Create and activate virtual environment, install dependencies, and run the server
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade --force-reinstall -r requirements.txt
python server.py

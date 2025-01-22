#!/bin/bash

rm -rf build/ dist/ *.egg-info .eggs/ __pycache__/ */__pycache__/
find . -type d -name "*.egg-info" -exec rm -r {} +
find . -type d -name "__pycache__" -exec rm -r {} +

python -m pip install --upgrade pip build twine setuptools>=61.0.0

python -m build

python -m twine check dist/*

if [ "$1" = "--upload" ]; then
    python -m twine upload dist/*
fi
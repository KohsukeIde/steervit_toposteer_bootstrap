#!/bin/bash

printf "\033[0;32m Launching isort \033[0m\n"
python -m isort scripts franca rasa  ./*.py

printf "\033[0;32m Launching black \033[0m\n"
python -m black --include=".*\.(py|ipynb)$" scripts franca rasa  ./*.py

printf "\033[0;32m Launching flake8 \033[0m\n"
python -m flake8 scripts franca rasa  ./*.py

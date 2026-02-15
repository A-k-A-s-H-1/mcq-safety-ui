#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y python3.10 python3.10-dev python3.10-distutils cmake pkg-config build-essential

# Use python3.10 explicitly
python3.10 -m pip install --upgrade pip
python3.10 -m pip install -r requirements.txt

#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update and install dependencies
echo "Updating system and installing necessary dependencies..."
sudo apt-get update
sudo apt-get install -y curl wget python3-pip

# Install ollama if not already installed
if ! command -v ollama &> /dev/null
then
    echo "Installing Ollama..."
    curl -o- https://ollama.com/install.sh | bash
else
    echo "Ollama is already installed."
fi

# Pull the Llama 3.2 model
echo "Pulling Llama 3.2 model..."
ollama pull llama3.2

# Install Python dependencies if requirements.txt is present
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found."
fi

echo "Setup completed successfully!"

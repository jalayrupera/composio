#!/bin/bash

# Remove ~/.composio folder if it exists
echo "Removing ~/.composio folder if it exists..."
if [ -d ~/.composio ]; then
    rm -rf ~/.composio
    echo "~/.composio folder removed."
else
    echo "~/.composio folder not found, skipping removal."
fi

# Create a virtual environment
echo "Creating virtual environment..."
python3.10 -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install libraries from requirements.txt
echo "Installing libraries from requirements.txt..."
pip install -r requirements.txt

# Find the composio root directory by navigating up the directory tree
# Current script location: /examples/quickstarters/document_qa_agent
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSIO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Define paths for local packages based on the discovered root
COMPOSIO_CORE_PATH="$COMPOSIO_ROOT"
COMPOSIO_CREW_PATH="$COMPOSIO_ROOT/plugins/crew_ai"

# Verify paths exist
if [ ! -d "$COMPOSIO_CORE_PATH" ]; then
    echo "Error: Local composio package directory not found at $COMPOSIO_CORE_PATH"
    exit 1
fi

if [ ! -d "$COMPOSIO_CREW_PATH" ]; then
    echo "Error: Local composio-crewai plugin directory not found at $COMPOSIO_CREW_PATH"
    exit 1
fi

# Install local composio core package
echo "Installing local composio core package from $COMPOSIO_CORE_PATH"
pip install -e "$COMPOSIO_CORE_PATH"

# Install local composio-crewai plugin
echo "Installing local composio-crewai plugin from $COMPOSIO_CREW_PATH"
pip install -e "$COMPOSIO_CREW_PATH"

# Copy env backup to .env file
if [ -f ".env.example" ]; then
    echo "Copying .env.example to .env..."
    cp .env.example .env
else
    echo "No .env.example file found. Creating a new .env file..."
    touch .env
    echo "# Add your environment variables below" > .env
fi

echo "Please fill in the .env file with the necessary environment variables."

echo "Setup completed successfully! The environment includes local development versions of composio and composio-crewai."
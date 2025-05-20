#!/bin/bash

echo "Running pre-deployment setup..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python is not installed. Please install Python and try again."
    exit 1
fi

# Run the setup_deployment.py script
python3 setup_deployment.py

if [ $? -ne 0 ]; then
    echo "Failed to run setup_deployment.py. Please check the error message above."
    exit 1
fi

echo "Pre-deployment setup completed successfully."
echo "You can now deploy your application."
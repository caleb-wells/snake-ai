#!/usr/bin/env bash

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print step message with formatting
print_step() {
    echo -e "\n${YELLOW}=== $1 ===${NC}"
}

# Print success message
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Print error message and exit
print_error() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

# Check Python version
check_python_version() {
    local min_version="3.8"
    local python_version
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

    if ! python3 -c "import sys; assert sys.version_info >= tuple(map(int, '${min_version}'.split('.')))"; then
        print_error "Python ${min_version} or higher is required. Found: ${python_version}"
    fi
}

# Main setup function
main() {
    print_step "Checking Python version"
    check_python_version
    print_success "Python version check passed"

    # Create and activate virtual environment
    print_step "Setting up Python virtual environment"
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_success "Virtual environment already exists"
    fi

    # Activate virtual environment
    source venv/bin/activate || print_error "Failed to activate virtual environment"
    print_success "Virtual environment activated"

    # Upgrade pip
    print_step "Upgrading pip"
    python3 -m pip install --upgrade pip
    print_success "Pip upgraded"

    # Install dependencies
    print_step "Installing project dependencies"
    python3 -m pip install -r requirements.txt
    print_success "Project dependencies installed"

    # Install development tools
    print_step "Installing development tools"
    python3 -m pip install pre-commit
    print_success "Pre-commit installed"

    # Setup pre-commit hooks
    print_step "Setting up pre-commit hooks"
    pre-commit install
    print_success "Pre-commit hooks installed"

    print_step "Environment setup complete!"
    echo -e "\nTo get started:"
    echo "1. The virtual environment is now active"
    echo "2. Run the training script: python src/main.py"
    echo "3. To deactivate the environment: deactivate"
    echo "4. To reactivate later: source venv/bin/activate"
}

# Run main function
main

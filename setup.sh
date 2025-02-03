#!/usr/bin/env bash

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print step message with formatting
print_step() {
    echo -e "${YELLOW}=== $1 ===${NC}"
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

# Check if command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is not installed. Please install it first."
    fi
}

# Check Python version
check_python_version() {
    local min_version="3.8"
    local python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

    if ! python3 -c "import sys; assert sys.version_info >= tuple(map(int, '${min_version}'.split('.')))"; then
        print_error "Python ${min_version} or higher is required. Found: ${python_version}"
    fi
}

# Main setup function
main() {
    print_step "Checking requirements"

    # Check for required commands
    check_command "python3"
    check_command "pip3"
    check_command "git"

    # Check Python version
    check_python_version
    print_success "All requirements satisfied"

    # Create directories if they don't exist
    print_step "Creating project structure"
    mkdir -p src/{game,ai,utils} tests notebooks data logs checkpoints
    touch src/{game,ai,utils}/__init__.py
    touch src/__init__.py
    print_success "Project directories created"

    # Initialize git if not already initialized
    if [ ! -d .git ]; then
        print_step "Initializing git repository"
        git init
        print_success "Git repository initialized"
    fi

    # Create virtual environment
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
    python3 -m pip install --upgrade pip || print_error "Failed to upgrade pip"
    print_success "Pip upgraded"

    # Install build dependencies first
    print_step "Installing build dependencies"
    python3 -m pip install --upgrade build hatchling || print_error "Failed to install build dependencies"
    print_success "Build dependencies installed"

    # Install dependencies
    print_step "Installing project dependencies"
    python3 -m pip install -e ".[dev,profiling,docs]" || print_error "Failed to install dependencies"
    print_success "Project dependencies installed"

    # Install and setup pre-commit
    print_step "Setting up pre-commit hooks"
    python3 -m pip install pre-commit || print_error "Failed to install pre-commit"
    pre-commit install || print_error "Failed to install pre-commit hooks"
    print_success "Pre-commit hooks installed"

    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        print_step "Creating .env file"
        cat > .env << EOL
# Environment Configuration
DEBUG=True
LOG_LEVEL=INFO

# Training Configuration
BATCH_SIZE=64
LEARNING_RATE=0.001
GAMMA=0.99
EPSILON_START=1.0
EPSILON_END=0.01
EPSILON_DECAY=0.995

# Game Configuration
WINDOW_SIZE=400
GRID_SIZE=20
FPS=30

# Paths
CHECKPOINT_DIR=checkpoints
LOG_DIR=logs
EOL
        print_success ".env file created"
    fi

    # Run pre-commit on all files
    print_step "Running pre-commit checks"
    pre-commit run --all-files || true

    print_step "Setup complete!"
    echo -e "\nNext steps:"
    echo "1. Your virtual environment is now active. To deactivate, run: deactivate"
    echo "2. To reactivate later, run: source venv/bin/activate"
    echo "3. Start coding! The project structure is ready."
    echo "4. Check the README.md for more information."
}

# Run main function
main

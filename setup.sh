#!/bin/bash

set -e

echo "============================================"
echo "  Sora MCP Server Setup"
echo "============================================"
echo ""

# Check if .env already exists
if [ -f .env ]; then
    echo "⚠️  .env file already exists!"
    read -p "Do you want to overwrite it? (y/N): " overwrite
    if [[ ! $overwrite =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. Your existing .env file was not modified."
        exit 0
    fi
    echo ""
fi

# Check for OPENAI_API_KEY in environment
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✓ Found OPENAI_API_KEY in environment"
    DEFAULT_API_KEY="$OPENAI_API_KEY"
    USE_ENV_KEY=true
else
    echo "ℹ️  No OPENAI_API_KEY found in environment"
    USE_ENV_KEY=false
fi

echo ""
echo "============================================"
echo "  Configuration"
echo "============================================"
echo ""

# Get API Key
if [ "$USE_ENV_KEY" = true ]; then
    read -p "Use existing OPENAI_API_KEY from environment? (Y/n): " use_existing
    if [[ $use_existing =~ ^[Nn]$ ]]; then
        read -s -p "Enter your OpenAI API key: " API_KEY
        echo ""
    else
        API_KEY="$DEFAULT_API_KEY"
        echo "Using API key from environment"
    fi
else
    read -s -p "Enter your OpenAI API key: " API_KEY
    echo ""
fi

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get video path
echo ""
echo "Default video download path: $PROJECT_DIR/sora-videos"
read -p "Press Enter to accept, or type a custom path: " VIDEO_PATH
if [ -z "$VIDEO_PATH" ]; then
    VIDEO_PATH="$PROJECT_DIR/sora-videos"
else
    # Convert to absolute path if relative
    if [[ "$VIDEO_PATH" != /* ]]; then
        VIDEO_PATH="$(cd "$(dirname "$VIDEO_PATH")" 2>/dev/null && pwd)/$(basename "$VIDEO_PATH")" || VIDEO_PATH="$PROJECT_DIR/$VIDEO_PATH"
    fi
fi

# Get reference path
echo ""
echo "Default reference images path: $PROJECT_DIR/reference-images"
read -p "Press Enter to accept, or type a custom path: " REFERENCE_PATH
if [ -z "$REFERENCE_PATH" ]; then
    REFERENCE_PATH="$PROJECT_DIR/reference-images"
else
    # Convert to absolute path if relative
    if [[ "$REFERENCE_PATH" != /* ]]; then
        REFERENCE_PATH="$(cd "$(dirname "$REFERENCE_PATH")" 2>/dev/null && pwd)/$(basename "$REFERENCE_PATH")" || REFERENCE_PATH="$PROJECT_DIR/$REFERENCE_PATH"
    fi
fi

echo ""
echo "============================================"
echo "  Creating directories..."
echo "============================================"

# Create directories
mkdir -p "$VIDEO_PATH"
echo "✓ Created $VIDEO_PATH"

mkdir -p "$REFERENCE_PATH"
echo "✓ Created $REFERENCE_PATH"

echo ""
echo "============================================"
echo "  Writing .env file..."
echo "============================================"

# Write .env file
cat > .env << EOF
OPENAI_API_KEY="$API_KEY"
SORA_VIDEO_PATH="$VIDEO_PATH"
REFERENCE_IMAGE_PATH="$REFERENCE_PATH"
EOF

echo "✓ Created .env file"

echo ""
echo "============================================"
echo "  Installing dependencies..."
echo "============================================"

# Run uv sync
if command -v uv &> /dev/null; then
    uv sync --dev
    echo "✓ Dependencies installed"
else
    echo "⚠️  'uv' command not found. Please install uv and run 'uv sync' manually."
    echo "   Visit: https://github.com/astral-sh/uv"
fi

echo ""
echo "============================================"
echo "  Setup Complete! 🎉"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Run 'claude' to start Claude Code"
echo "  2. The Sora MCP server will connect automatically"
echo "  3. Start generating videos!"
echo ""
echo "Configuration saved to .env:"
echo "  - Videos will be saved to: $VIDEO_PATH"
echo "  - Reference images: $REFERENCE_PATH"
echo ""

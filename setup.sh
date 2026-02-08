#!/bin/bash

set -e

echo "============================================"
echo "  sanzaru Setup"
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

# Get media path
echo ""
echo "Default media path: $PROJECT_DIR/media"
read -p "Press Enter to accept, or type a custom path: " MEDIA_PATH
if [ -z "$MEDIA_PATH" ]; then
    MEDIA_PATH="$PROJECT_DIR/media"
else
    # Convert to absolute path if relative
    if [[ "$MEDIA_PATH" != /* ]]; then
        MEDIA_PATH="$(cd "$(dirname "$MEDIA_PATH")" 2>/dev/null && pwd)/$(basename "$MEDIA_PATH")" || MEDIA_PATH="$PROJECT_DIR/$MEDIA_PATH"
    fi
fi

echo ""
echo "============================================"
echo "  Creating directories..."
echo "============================================"

# Create media directory structure
mkdir -p "$MEDIA_PATH/videos" "$MEDIA_PATH/images" "$MEDIA_PATH/audio"
echo "✓ Created $MEDIA_PATH/videos"
echo "✓ Created $MEDIA_PATH/images"
echo "✓ Created $MEDIA_PATH/audio"

echo ""
echo "============================================"
echo "  Writing .env file..."
echo "============================================"

# Write .env file
cat > .env << EOF
OPENAI_API_KEY="$API_KEY"
SANZARU_MEDIA_PATH="$MEDIA_PATH"
EOF

echo "✓ Created .env file"

echo ""
echo "============================================"
echo "  Installing dependencies..."
echo "============================================"

# Run uv sync
if command -v uv &> /dev/null; then
    uv sync --all-extras --dev
    echo "✓ Dependencies installed"
else
    echo "⚠️  'uv' command not found. Please install uv and run 'uv sync --all-extras --dev' manually."
    echo "   Visit: https://github.com/astral-sh/uv"
fi

echo ""
echo "============================================"
echo "  Setup Complete! 🎉"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Run 'claude' to start Claude Code"
echo "  2. The sanzaru MCP server will connect automatically"
echo "  3. Start generating videos!"
echo ""
echo "Configuration saved to .env:"
echo "  - Media root: $MEDIA_PATH"
echo "  - Videos: $MEDIA_PATH/videos"
echo "  - Images: $MEDIA_PATH/images"
echo "  - Audio: $MEDIA_PATH/audio"
echo ""

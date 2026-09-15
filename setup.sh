#!/bin/bash

set -e

# .env values are written inside double quotes. Escape the two characters that
# would otherwise break out of the quoted string: a bare " ends the value early
# and leaves the rest of the line as garbage (python-dotenv drops the whole
# binding), and a trailing \ would escape the closing quote. Backslash must be
# substituted first, or it would double the backslashes added after it.
# Deliberately NOT escaped: $ and ` — .env is not shell, and python-dotenv only
# decodes \\ \' \" \a \b \f \n \r \t \v, so a \$ would survive as a literal
# backslash in the value. (A path containing the braced form ${VAR} is still
# interpolated by python-dotenv; its grammar offers no escape for that.)
env_escape() {
    local value=$1
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    printf '%s' "$value"
}

echo "============================================"
echo "  sanzaru Setup"
echo "============================================"
echo ""

# A symlink at ./.env is refused outright rather than written through. Every
# step below — the truncate, the chmod, the heredoc — would otherwise land the
# API key in whatever the link names, and `-f` is false for a dangling link, so
# the overwrite prompt would not even fire. The check is `-L`, which does not
# follow the link; `-e` below does, so the prompt still covers a plain file.
if [ -L .env ]; then
    echo "❌ .env is a symbolic link — refusing to write the API key through it."
    echo "   Remove the link (or run setup from a different directory) and try again."
    exit 1
fi

# Check if .env already exists
if [ -e .env ]; then
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

# Write .env file.
# The API key is stored here in cleartext, so the file must never be readable by
# other local users. Create it owner-only *before* the secret lands in it: the
# umask covers a newly created .env, and the chmod covers a pre-existing one,
# because `>` truncates a file but leaves its old mode alone. Do not collapse
# these into a bare `cat > .env` — on the default umask 022 that yields 0644.
(umask 077 && : > .env)
chmod 600 .env

cat > .env << EOF
OPENAI_API_KEY="$(env_escape "$API_KEY")"
SANZARU_MEDIA_PATH="$(env_escape "$MEDIA_PATH")"
EOF

echo "✓ Created .env file (permissions: 600)"

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

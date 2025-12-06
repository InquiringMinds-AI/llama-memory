#!/bin/bash
#
# llama-memory installation script
# Builds llama.cpp, downloads embedding model, and installs sqlite-vec
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# XDG directories
XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
DATA_DIR="$XDG_DATA_HOME/llama-memory"
CACHE_DIR="$XDG_CACHE_HOME/llama-memory"

# Versions
SQLITE_VEC_VERSION="0.1.6"
EMBEDDING_MODEL_URL="https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q4_K_M.gguf"
EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2-Q4_K_M.gguf"

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)
IS_TERMUX=false
if [[ "$HOME" == *"com.termux"* ]]; then
    IS_TERMUX=true
fi

info "Detected: OS=$OS ARCH=$ARCH TERMUX=$IS_TERMUX"

# Create directories
mkdir -p "$DATA_DIR"/{bin,lib,models}
mkdir -p "$CACHE_DIR"

#
# Step 1: Install build dependencies
#
info "Checking build dependencies..."

if $IS_TERMUX; then
    pkg install -y cmake make git curl 2>/dev/null || true
else
    if ! command -v cmake &> /dev/null; then
        error "cmake not found. Install it with your package manager."
    fi
    if ! command -v make &> /dev/null; then
        error "make not found. Install it with your package manager."
    fi
fi

#
# Step 2: Build llama.cpp
#
LLAMA_CPP_DIR="$CACHE_DIR/llama.cpp"

if [[ -f "$DATA_DIR/bin/llama-embedding" ]]; then
    info "llama-embedding already installed, skipping build"
else
    info "Building llama.cpp (this may take a few minutes)..."

    if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
        git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
    fi

    cd "$LLAMA_CPP_DIR"

    mkdir -p build
    cd build

    # Configure - disable OpenMP on Termux to avoid issues
    if $IS_TERMUX; then
        cmake .. -DGGML_OPENMP=OFF
    else
        cmake ..
    fi

    # Build just what we need
    make -j$(nproc) llama-embedding 2>&1 | tail -20

    # Copy binaries and libraries
    cp bin/llama-embedding "$DATA_DIR/bin/"
    cp bin/lib*.so* "$DATA_DIR/lib/" 2>/dev/null || true

    info "llama-embedding installed to $DATA_DIR/bin/"
fi

#
# Step 3: Download embedding model
#
MODEL_PATH="$DATA_DIR/models/$EMBEDDING_MODEL_NAME"

if [[ -f "$MODEL_PATH" ]]; then
    info "Embedding model already downloaded"
else
    info "Downloading embedding model (~21MB)..."
    curl -L -o "$MODEL_PATH" "$EMBEDDING_MODEL_URL"
    info "Model downloaded to $MODEL_PATH"
fi

#
# Step 4: Install sqlite-vec
#
if [[ -f "$DATA_DIR/lib/vec0.so" ]]; then
    info "sqlite-vec already installed"
else
    info "Installing sqlite-vec..."

    # Determine platform string
    if $IS_TERMUX || [[ "$OS" == "Linux" && "$ARCH" == "aarch64" ]]; then
        SQLITE_VEC_PLATFORM="android-aarch64"
    elif [[ "$OS" == "Linux" && "$ARCH" == "x86_64" ]]; then
        SQLITE_VEC_PLATFORM="linux-x86_64"
    elif [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
        SQLITE_VEC_PLATFORM="macos-aarch64"
    elif [[ "$OS" == "Darwin" && "$ARCH" == "x86_64" ]]; then
        SQLITE_VEC_PLATFORM="macos-x86_64"
    else
        error "Unsupported platform: $OS $ARCH"
    fi

    SQLITE_VEC_URL="https://github.com/asg017/sqlite-vec/releases/download/v${SQLITE_VEC_VERSION}/sqlite-vec-${SQLITE_VEC_VERSION}-loadable-${SQLITE_VEC_PLATFORM}.tar.gz"

    curl -L -o "$CACHE_DIR/sqlite-vec.tar.gz" "$SQLITE_VEC_URL"
    tar -xzf "$CACHE_DIR/sqlite-vec.tar.gz" -C "$DATA_DIR/lib/"
    rm "$CACHE_DIR/sqlite-vec.tar.gz"

    info "sqlite-vec installed to $DATA_DIR/lib/"
fi

#
# Step 5: Verify installation
#
info "Verifying installation..."

# Test embedding binary
export LD_LIBRARY_PATH="$DATA_DIR/lib:$LD_LIBRARY_PATH"
if "$DATA_DIR/bin/llama-embedding" -m "$MODEL_PATH" -p "test" --log-disable 2>&1 | grep -q "embedding 0:"; then
    info "Embedding generation: OK"
else
    warn "Embedding generation test failed"
fi

# Test sqlite-vec
if python3 -c "import sqlite3; c = sqlite3.connect(':memory:'); c.enable_load_extension(True); c.load_extension('$DATA_DIR/lib/vec0'); print(c.execute('SELECT vec_version()').fetchone()[0])" 2>/dev/null; then
    info "sqlite-vec: OK"
else
    warn "sqlite-vec test failed"
fi

#
# Done
#
echo
info "Installation complete!"
echo
echo "Data directory: $DATA_DIR"
echo "  - bin/llama-embedding"
echo "  - lib/vec0.so"
echo "  - lib/libllama.so (and others)"
echo "  - models/$EMBEDDING_MODEL_NAME"
echo
echo "Next steps:"
echo "  1. Install llama-memory: pip install -e /path/to/llama-memory"
echo "  2. Initialize database: llama-memory init"
echo "  3. Check health: llama-memory doctor"
echo

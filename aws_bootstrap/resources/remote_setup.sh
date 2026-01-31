#!/usr/bin/env bash
# remote_setup.sh — Post-boot setup for Deep Learning AMI instances.
# Runs on the EC2 instance after SSH becomes available.
set -euo pipefail

echo "=== aws-bootstrap-g4dn remote setup ==="

# 1. Verify GPU
echo ""
echo "[1/6] Verifying GPU and CUDA..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found"
fi

if command -v nvcc &>/dev/null; then
    nvcc --version | grep "release"
else
    echo "WARNING: nvcc not found (CUDA toolkit may not be installed)"
fi

# Make Nsight Systems (nsys) available on PATH if installed under /opt/nvidia
if ! command -v nsys &>/dev/null; then
    NSIGHT_DIR="/opt/nvidia/nsight-systems"
    if [ -d "$NSIGHT_DIR" ]; then
        # Fix permissions — the parent dir is often root-only (drwx------)
        sudo chmod o+rx "$NSIGHT_DIR"
        # Find the latest version directory (lexicographic sort)
        NSYS_VERSION=$(ls -1 "$NSIGHT_DIR" | sort -V | tail -1)
        if [ -n "$NSYS_VERSION" ] && [ -x "$NSIGHT_DIR/$NSYS_VERSION/bin/nsys" ]; then
            NSYS_BIN="$NSIGHT_DIR/$NSYS_VERSION/bin"
            if ! grep -q "nsight-systems" ~/.bashrc 2>/dev/null; then
                echo "export PATH=\"$NSYS_BIN:\$PATH\"" >> ~/.bashrc
            fi
            export PATH="$NSYS_BIN:$PATH"
            echo "  Nsight Systems $NSYS_VERSION added to PATH ($NSYS_BIN)"
        else
            echo "  WARNING: Nsight Systems directory found but no nsys binary"
        fi
    else
        echo "  Nsight Systems not found at $NSIGHT_DIR"
    fi
else
    echo "  nsys already on PATH: $(command -v nsys)"
fi

# 2. Install utilities
echo ""
echo "[2/6] Installing utilities..."
sudo apt-get update -qq
sudo apt-get install -y -qq htop tmux tree jq

# 3. Set up Python environment with uv
echo ""
echo "[3/6] Setting up Python environment with uv..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ -n "${PYTHON_VERSION:-}" ]; then
    echo "  Installing Python ${PYTHON_VERSION}..."
    uv python install "$PYTHON_VERSION"
    uv venv --python "$PYTHON_VERSION" ~/venv
else
    uv venv ~/venv
fi

# --- CUDA-aware PyTorch installation ---
# Known PyTorch CUDA wheel tags (ascending order).
# Update this list when PyTorch publishes new CUDA builds.
# See: https://download.pytorch.org/whl/
KNOWN_CUDA_TAGS=(118 121 124 126 128 129 130)

detect_cuda_version() {
    # Primary: nvcc (actual toolkit installed on the system)
    if command -v nvcc &>/dev/null; then
        nvcc --version | grep -oP 'release \K[\d.]+'
        return
    fi
    # Fallback: nvidia-smi (max CUDA the driver supports)
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+'
        return
    fi
    echo ""
}

cuda_version_to_tag() {
    # "12.9" → "129", "13.0" → "130"
    echo "$1" | tr -d '.'
}

find_best_cuda_tag() {
    local detected_tag="$1"
    local best=""
    for tag in "${KNOWN_CUDA_TAGS[@]}"; do
        if [ "$tag" -le "$detected_tag" ]; then
            best="$tag"
        fi
    done
    echo "$best"
}

install_pytorch_cuda() {
    local cuda_ver
    cuda_ver=$(detect_cuda_version)

    if [ -z "$cuda_ver" ]; then
        echo "  WARNING: No CUDA detected — installing PyTorch from PyPI (CPU or default CUDA)"
        uv pip install --python ~/venv/bin/python torch torchvision
        return
    fi
    echo "  Detected CUDA version: $cuda_ver"

    local detected_tag
    detected_tag=$(cuda_version_to_tag "$cuda_ver")

    local best_tag
    best_tag=$(find_best_cuda_tag "$detected_tag")

    if [ -z "$best_tag" ]; then
        echo "  WARNING: No matching PyTorch CUDA tag for cu${detected_tag} — installing from PyPI"
        uv pip install --python ~/venv/bin/python torch torchvision
        return
    fi

    echo "  Using PyTorch CUDA index: cu${best_tag}"
    if ! uv pip install --python ~/venv/bin/python \
            --default-index "https://download.pytorch.org/whl/cu${best_tag}" \
            torch torchvision; then
        echo "  WARNING: CUDA index install failed — falling back to PyPI"
        uv pip install --python ~/venv/bin/python torch torchvision
    fi
}

install_pytorch_cuda

# Install remaining dependencies (torch/torchvision already installed above)
uv pip install --python ~/venv/bin/python -r /tmp/requirements.txt

# Copy GPU benchmark script and smoke test notebook
cp /tmp/gpu_benchmark.py ~/gpu_benchmark.py
cp /tmp/gpu_smoke_test.ipynb ~/gpu_smoke_test.ipynb

# Auto-activate venv on login
if ! grep -q 'source ~/venv/bin/activate' ~/.bashrc 2>/dev/null; then
    echo 'source ~/venv/bin/activate' >> ~/.bashrc
fi

# Quick CUDA smoke test
echo "  Running CUDA smoke test..."
if ~/venv/bin/python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
x = torch.randn(256, 256, device='cuda')
y = torch.mm(x, x)
torch.cuda.synchronize()
print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
print('  Quick matmul test: PASSED')
"; then
    echo "  CUDA smoke test passed"
else
    echo "  WARNING: CUDA smoke test failed — check PyTorch/CUDA installation"
fi

JUPYTER_CONFIG_DIR="$HOME/.jupyter"
mkdir -p "$JUPYTER_CONFIG_DIR"
cat > "$JUPYTER_CONFIG_DIR/jupyter_lab_config.py" << 'PYEOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.IdentityProvider.token = ''
c.ServerApp.allow_remote_access = True
PYEOF
echo "  Jupyter config written to $JUPYTER_CONFIG_DIR/jupyter_lab_config.py"

# 4. Jupyter systemd service
echo ""
echo "[4/6] Setting up Jupyter systemd service..."
LOGIN_USER=$(whoami)

sudo tee /etc/systemd/system/jupyter.service > /dev/null << SVCEOF
[Unit]
Description=Jupyter Lab Server
After=network.target

[Service]
Type=simple
User=${LOGIN_USER}
WorkingDirectory=/home/${LOGIN_USER}
ExecStart=/home/${LOGIN_USER}/venv/bin/python -m jupyterlab
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
SVCEOF

sudo systemctl daemon-reload
sudo systemctl enable jupyter.service
sudo systemctl start jupyter.service
echo "  Jupyter service started (port 8888)"

# 5. SSH keepalive
echo ""
echo "[5/6] Configuring SSH keepalive..."
if ! grep -q "ClientAliveInterval" /etc/ssh/sshd_config; then
    echo "ClientAliveInterval 60" | sudo tee -a /etc/ssh/sshd_config > /dev/null
    echo "ClientAliveCountMax 10" | sudo tee -a /etc/ssh/sshd_config > /dev/null
    sudo systemctl reload sshd
    echo "  SSH keepalive configured"
else
    echo "  SSH keepalive already configured"
fi

# 6. VSCode workspace setup
echo ""
echo "[6/6] Setting up VSCode workspace..."
mkdir -p ~/workspace/.vscode

# Detect cuda-gdb path
CUDA_GDB_PATH=""
if command -v cuda-gdb &>/dev/null; then
    CUDA_GDB_PATH=$(command -v cuda-gdb)
elif [ -x /usr/local/cuda/bin/cuda-gdb ]; then
    CUDA_GDB_PATH="/usr/local/cuda/bin/cuda-gdb"
else
    # Try glob for versioned CUDA installs
    for p in /usr/local/cuda-*/bin/cuda-gdb; do
        if [ -x "$p" ]; then
            CUDA_GDB_PATH="$p"
        fi
    done
fi
if [ -z "$CUDA_GDB_PATH" ]; then
    echo "  WARNING: cuda-gdb not found — using placeholder in launch.json"
    CUDA_GDB_PATH="cuda-gdb"
else
    echo "  cuda-gdb: $CUDA_GDB_PATH"
fi

# Detect GPU SM architecture
GPU_ARCH=""
if command -v nvidia-smi &>/dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
    if [ -n "$COMPUTE_CAP" ]; then
        GPU_ARCH="sm_$(echo "$COMPUTE_CAP" | tr -d '.')"
    fi
fi
if [ -z "$GPU_ARCH" ]; then
    echo "  WARNING: Could not detect GPU arch — defaulting to sm_75"
    GPU_ARCH="sm_75"
else
    echo "  GPU arch: $GPU_ARCH"
fi

# Copy example CUDA source into workspace
cp /tmp/saxpy.cu ~/workspace/saxpy.cu
echo "  Deployed saxpy.cu"

# Deploy launch.json with cuda-gdb path
sed "s|__CUDA_GDB_PATH__|${CUDA_GDB_PATH}|g" /tmp/launch.json > ~/workspace/.vscode/launch.json
echo "  Deployed launch.json"

# Deploy tasks.json with GPU architecture
sed "s|__GPU_ARCH__|${GPU_ARCH}|g" /tmp/tasks.json > ~/workspace/.vscode/tasks.json
echo "  Deployed tasks.json"

echo ""
echo "=== Remote setup complete ==="

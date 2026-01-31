#!/usr/bin/env bash
# remote_setup.sh — Post-boot setup for Deep Learning AMI instances.
# Runs on the EC2 instance after SSH becomes available.
set -euo pipefail

echo "=== aws-bootstrap-g4dn remote setup ==="

# 1. Verify GPU
echo ""
echo "[1/5] Verifying GPU and CUDA..."
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

# 2. Install utilities
echo ""
echo "[2/5] Installing utilities..."
sudo apt-get update -qq
sudo apt-get install -y -qq htop tmux tree jq

# 3. Set up Python environment with uv
echo ""
echo "[3/5] Setting up Python environment with uv..."
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
echo "[4/5] Setting up Jupyter systemd service..."
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
echo "[5/5] Configuring SSH keepalive..."
if ! grep -q "ClientAliveInterval" /etc/ssh/sshd_config; then
    echo "ClientAliveInterval 60" | sudo tee -a /etc/ssh/sshd_config > /dev/null
    echo "ClientAliveCountMax 10" | sudo tee -a /etc/ssh/sshd_config > /dev/null
    sudo systemctl reload sshd
    echo "  SSH keepalive configured"
else
    echo "  SSH keepalive already configured"
fi

echo ""
echo "=== Remote setup complete ==="

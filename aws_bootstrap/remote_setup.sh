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

# 3. Configure Jupyter
echo ""
echo "[3/5] Configuring Jupyter Lab..."
if ! command -v jupyter &>/dev/null; then
    pip install --quiet jupyterlab
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
JUPYTER_BIN=$(command -v jupyter || echo "/usr/local/bin/jupyter")
LOGIN_USER=$(whoami)

sudo tee /etc/systemd/system/jupyter.service > /dev/null << SVCEOF
[Unit]
Description=Jupyter Lab Server
After=network.target

[Service]
Type=simple
User=${LOGIN_USER}
WorkingDirectory=/home/${LOGIN_USER}
ExecStart=${JUPYTER_BIN} lab
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

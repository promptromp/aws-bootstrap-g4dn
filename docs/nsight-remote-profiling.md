# NVIDIA Nsight Remote GPU Profiling on EC2

Guide to using NVIDIA's Nsight profiling and debugging tools with remote EC2 GPU instances provisioned by `aws-bootstrap`.

## Overview

NVIDIA provides several Nsight tools for GPU profiling and debugging. The most relevant ones for remote EC2 work are:

| Tool | Purpose | macOS Host | Ports Required | Best Approach |
|------|---------|-----------|----------------|---------------|
| **Nsight Compute** | CUDA kernel profiling | Native GUI | SSH only (22) | GUI remote or CLI + local viewer |
| **Nsight Systems** | System-wide tracing | Native GUI | SSH (22) + 45555 | CLI + local viewer |
| **Nsight VSCE** | Interactive CUDA debugging | Via VSCode | SSH only (22) | VSCode Remote SSH |
| **Nsight Graphics** | Graphics/shader profiling | No | SSH only (22) | CLI captures (graphics workloads only) |

---

## Nsight Compute (Kernel-Level Profiler)

Nsight Compute is the most straightforward tool for remote profiling over SSH. It provides per-kernel performance metrics, roofline analysis, occupancy analysis, and memory throughput data.

### How It Works

The GUI (`ncu-ui`) runs on your local machine and connects to the EC2 instance over SSH. Nsight Compute automatically deploys its CLI tools to a deployment directory on the remote target on first connection. All profiling traffic is tunneled through SSH — no extra ports needed.

Two profiling modes are available:

- **Interactive:** A SOCKS proxy tunnels through SSH, letting you step through kernels and control execution in real time.
- **Non-Interactive:** The profiler runs to completion on the remote and copies the report back automatically via SSH remote forwarding.

### Setup

**Local machine (macOS/Linux/Windows):**

1. Download Nsight Compute from [NVIDIA Developer](https://developer.nvidia.com/tools-overview/nsight-compute/get-started) (free, requires NVIDIA developer account)
2. Install `ncu-ui` (the GUI application). As of 2025, macOS ARM64 (Apple Silicon) is natively supported.

**Remote EC2 instance:**

Nothing extra is needed — the GUI auto-deploys the CLI on first connection. The Deep Learning AMI already includes the CUDA toolkit.

### GPU Performance Counter Permissions

By default, non-admin users cannot access GPU performance counters, which results in `ERR_NVGPUCTRPERM` errors. To fix this:

```bash
ssh aws-gpu1
sudo bash -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia.conf'
sudo update-initramfs -u -k all
sudo reboot
```

> **Important:** Rebooting an EC2 instance without an Elastic IP will assign a new public IP. After reboot, run `aws-bootstrap status` to see the new IP and update the SSH config alias. You may need to `aws-bootstrap terminate` and re-launch, or manually update `~/.ssh/config`. This is a one-time setup per instance.

### Workflow A: GUI Remote Profiling

1. Open `ncu-ui` locally.
2. Click **Connect** and add a new SSH connection:
   - **Host:** your EC2 public IP (from `aws-bootstrap status`)
   - **Username:** `ubuntu`
   - **Port:** 22 (or your custom `--ssh-port`)
   - **Authentication:** Private key (`~/.ssh/id_ed25519`)
3. Select the CUDA binary to profile on the remote machine.
4. Choose an output file location on your local machine.
5. Click **Launch** to start profiling.

Nsight Compute supports `ProxyJump` and `ProxyCommand` SSH options if you need to reach the instance through a bastion host.

### Workflow B: CLI on Remote, View Locally (Recommended)

This is the most reliable approach — avoids real-time connection issues:

```bash
# Profile on the remote instance
ssh aws-gpu1 'ncu -o /tmp/profile --set full ./my_cuda_app'

# Download the report
scp aws-gpu1:/tmp/profile.ncu-rep .

# Open locally in the GUI
ncu-ui profile.ncu-rep
```

For source-level correlation, compile with `nvcc --lineinfo`.

### References

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [How to Set Up Nsight Compute on EC2](https://tspeterkim.github.io/posts/nsight-setup-on-ec2) — step-by-step walkthrough with screenshots

---

## Nsight Systems (System-Wide Profiler)

Nsight Systems traces CPU activity, GPU workloads (CUDA, Vulkan), OS runtime, threading, memory transfers, and NVTX annotations on a unified timeline. Useful for understanding end-to-end application performance.

### Security Caveat

Unlike Nsight Compute, Nsight Systems uses SSH only for the initial connection. **Actual profiling data transfers over a raw, unencrypted TCP socket on port 45555.** NVIDIA explicitly warns against using this on untrusted networks.

For EC2, you can mitigate this by tunneling port 45555 through SSH:

```bash
ssh -L 45555:localhost:45555 aws-gpu1
```

Then configure the Nsight Systems GUI to connect to `localhost` instead of the remote IP.

### Setup

**Local machine:**

Download Nsight Systems from [NVIDIA Developer](https://developer.nvidia.com/nsight-systems/get-started). The GUI (`nsys-ui`) is available for macOS, Linux, and Windows.

**Remote EC2 instance:**

The `nsys` CLI is typically included with the CUDA toolkit on Deep Learning AMIs. Verify with:

```bash
ssh aws-gpu1 'nsys status -e'
```

Additionally, Netcat must be installed (required by the remote profiling daemon):

```bash
ssh aws-gpu1 'sudo apt-get install -y netcat'
```

### Port Requirements

If using GUI remote profiling (not the CLI workflow), you need **port 45555** open in the EC2 security group in addition to SSH. The current `aws-bootstrap` security group only opens SSH — you would need to manually add the rule via the AWS console or CLI, or use the SSH tunnel approach described above.

### Workflow: CLI on Remote, View Locally (Recommended)

This avoids the port 45555 requirement entirely:

```bash
# Profile on the remote instance
ssh aws-gpu1 'nsys profile --trace=cuda,nvtx --output=/tmp/report ./my_app'

# Download the report
scp aws-gpu1:/tmp/report.nsys-rep .

# Open locally in the GUI
nsys-ui report.nsys-rep
```

### References

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Systems Installation Guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html)

---

## Nsight Visual Studio Code Edition (CUDA Debugger)

Nsight VSCE is a VS Code extension for building and debugging CUDA applications. This is the most natural fit for the `aws-bootstrap` workflow since it works directly with VSCode Remote SSH.

### How It Works

The extension provides CUDA debugging via `cuda-gdb` (or `cuda-gdbserver` for explicit remote setups). When used with VSCode Remote SSH, everything runs on the remote instance — the extension, the compiler, the debugger.

Features include:
- Breakpoints in GPU device code (including conditional breakpoints)
- GPU register, variable, and call-stack inspection
- Warp and lane focus controls (switch between streaming multiprocessors, warps, lanes)
- Full CPU thread inspection while stopped in GPU code, and vice versa
- CUDA-aware syntax highlighting and IntelliSense

### Setup

**Local machine:**

1. Install [VSCode](https://code.visualstudio.com/)
2. Install the [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension

**Remote EC2 instance (via VSCode Remote SSH):**

1. Connect to the instance: `code --folder-uri vscode-remote://ssh-remote+aws-gpu1/home/ubuntu/workspace`
2. Install the [Nsight VSCE extension](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition) on the remote (VS Code will prompt)
3. `cuda-gdb` is included with the CUDA toolkit on Deep Learning AMIs

### Debugging Workflow

1. Connect to `aws-gpu1` via VSCode Remote SSH (opens `~/workspace`).
2. `launch.json` and `tasks.json` are pre-configured in `~/workspace/.vscode/` with the detected `cuda-gdb` path and GPU architecture.
3. Open or create `.cu` files in `~/workspace`.
4. Set breakpoints in your `.cu` files.
5. Press F5 to start debugging.

### Known Issues

- `cuda-gdb` may require root privileges for GPU access. The same `NVreg_RestrictProfilingToAdminUsers=0` modprobe fix (described in the Nsight Compute section) resolves this. Alternatively, create a sudoers entry for `cuda-gdb`.
- Some users report the debugger failing to start on certain Remote SSH configurations. Check the Debug Console output for error details.

### References

- [Nsight VSCE Documentation](https://docs.nvidia.com/nsight-visual-studio-code-edition/latest/)
- [Nsight VSCE on VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition)
- [Nsight VSCE on GitHub](https://github.com/NVIDIA/nsight-vscode-edition)

---

## Quick Reference

### Common Setup: GPU Performance Counter Access

Required for Nsight Compute profiling and `cuda-gdb` debugging. This is a one-time setup per instance but **requires a reboot**:

```bash
ssh aws-gpu1
sudo bash -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia.conf'
sudo update-initramfs -u -k all
sudo reboot
```

After reboot, the instance will have a new public IP (unless using an Elastic IP). Run `aws-bootstrap status` to see the updated address.

### Recommended Approach: CLI Profiling + Local Viewer

The most practical and secure workflow for `aws-bootstrap` instances:

```bash
# Kernel profiling with Nsight Compute
ssh aws-gpu1 'ncu -o /tmp/profile --set full ./my_cuda_app'
scp aws-gpu1:/tmp/profile.ncu-rep .
ncu-ui profile.ncu-rep

# System profiling with Nsight Systems
ssh aws-gpu1 'nsys profile --trace=cuda,nvtx --output=/tmp/report ./my_app'
scp aws-gpu1:/tmp/report.nsys-rep .
nsys-ui report.nsys-rep
```

This requires no additional ports, no security group changes, and works with the existing SSH configuration that `aws-bootstrap` sets up.

### Port Summary

| Tool | Method | Ports |
|------|--------|-------|
| Nsight Compute (GUI remote) | SSH tunnel | 22 only |
| Nsight Compute (CLI + scp) | SSH | 22 only |
| Nsight Systems (GUI remote) | SSH + raw socket | 22 + 45555 |
| Nsight Systems (CLI + scp) | SSH | 22 only |
| Nsight VSCE (VSCode) | Remote SSH | 22 only |

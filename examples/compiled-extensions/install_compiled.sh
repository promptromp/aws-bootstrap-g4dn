#!/usr/bin/env bash
#
# Install Python packages that compile against the instance's PyTorch and CUDA
# (flash-attn, causal-conv1d, mamba-ssm, apex, ...) into ~/venv, with an
# optional S3 wheel cache so later instances skip the (often long) build.
#
# Run it on an instance after `aws-bootstrap launch`, from your machine:
#
#   ssh aws-gpu1 'bash -s -- causal-conv1d==1.5.0' < install_compiled.sh
#   ssh aws-gpu1 'WHEEL_CACHE=s3://my-bucket/wheels bash -s -- causal-conv1d==1.5.0' < install_compiled.sh
#
# ---------------------------------------------------------------------------
# WHY
# ---------------------------------------------------------------------------
# These packages import torch while they build, so they must be built with
# --no-build-isolation against the torch already in the venv; an isolated build
# installs some other torch (or fails), and a prebuilt wheel from PyPI rarely
# matches this instance's torch + CUDA. pip's own cache doesn't help a fresh
# instance, hence the S3 cache.
#
# A wheel's filename doesn't record the torch/CUDA it was built against, so
# cached wheels live under a key that does:
#
#   $WHEEL_CACHE/torch<version>-cp<python>-<arch>-sm<gpu>/   e.g. torch2.15.0_cu134-cp312-x86_64-sm89/
#
# A different torch, CUDA, Python, CPU architecture or GPU generation gets its
# own key, and so a fresh build.
#
# ---------------------------------------------------------------------------
# SETTINGS (environment variables)
# ---------------------------------------------------------------------------
# WHEEL_CACHE  s3://bucket/prefix to share wheels between instances (optional).
#              Reading needs s3:ListBucket + s3:GetObject, uploading s3:PutObject;
#              give the instance a role with `aws-bootstrap launch --iam-instance-profile`.
# MAX_JOBS     Parallel compile jobs (torch's extension builder honours it).
#              Lower it if the build runs out of memory.
# VENV         The venv to install into (default: ~/venv, made by remote setup).
#
# ---------------------------------------------------------------------------
# NOTES
# ---------------------------------------------------------------------------
# * A package pinned with == reuses only a cached wheel of that version; an
#   unpinned name reuses the newest cached version of it. Pin for reproducibility.
# * Build dependencies are not resolved: list packages that need compiling in
#   dependency order (causal-conv1d before mamba-ssm), so each is already
#   installed when the next one builds.
# * Safe to re-run: cached wheels install in seconds.
# * A failed cache read or upload only warns; the install itself must succeed.

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: install_compiled.sh PACKAGE[==VERSION] ..." >&2
    exit 2
fi

VENV="${VENV:-$HOME/venv}"
PY="$VENV/bin/python"
# A non-interactive ssh command doesn't read ~/.bashrc, so set up PATH here.
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"

if [ ! -x "$PY" ]; then
    echo "No venv at $VENV (did aws-bootstrap's remote setup run?)" >&2
    exit 1
fi

# The tools a non-isolated build expects to find in the venv.
uv pip install --python "$PY" --quiet pip setuptools wheel ninja packaging

torch_version=$("$PY" -c 'import torch; print(torch.__version__)')
py_tag=$("$PY" -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
gpu_sm=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
key="torch${torch_version//+/_}-${py_tag}-$(uname -m)-sm${gpu_sm}"
dir="$HOME/wheels/$key"
mkdir -p "$dir"
echo "[compiled] torch $torch_version; cache key $key"

if [ -n "${WHEEL_CACHE:-}" ]; then
    aws s3 sync "${WHEEL_CACHE%/}/$key/" "$dir/" --no-progress \
        || echo "[compiled] WARNING: could not read ${WHEEL_CACHE%/}/$key/; building what's missing" >&2
fi

# The newest wheel in $dir for a distribution name (and version, when given).
find_wheel() {
    local norm=$1 version=$2
    local -a matches
    # Wheel filenames normalise the name (runs of - _ . become _), and older
    # tools keep its case, so match case-insensitively. A local version
    # (1.5.0+cu12) follows the version with '+'.
    shopt -s nullglob nocaseglob
    if [ -n "$version" ]; then
        matches=("$dir/$norm-$version"-*.whl "$dir/$norm-$version"+*.whl)
    else
        matches=("$dir/$norm"-*.whl)
    fi
    shopt -u nullglob nocaseglob
    [ "${#matches[@]}" -gt 0 ] && printf '%s\n' "${matches[@]}" | sort -V | tail -n1
    return 0
}

built=0
for spec in "$@"; do
    name="${spec%%[=<>!~;[ ]*}"
    norm=$(printf '%s' "$name" | tr -s -- '-_.' '_')
    version=""
    if [[ "$spec" == *==* ]]; then
        version="${spec#*==}"
        version="${version%%[ ;,]*}"
    fi

    wheel=$(find_wheel "$norm" "$version")
    if [ -n "$wheel" ]; then
        echo "[compiled] $spec: using cached $(basename "$wheel")"
    else
        echo "[compiled] $spec: building against torch $torch_version (this can take a while)..."
        "$PY" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$dir" "$spec"
        wheel=$(find_wheel "$norm" "$version")
        if [ -z "$wheel" ]; then
            echo "[compiled] $spec: built, but found no matching wheel in $dir" >&2
            exit 1
        fi
        built=1
    fi
    uv pip install --python "$PY" "$wheel"
done

if [ "$built" -eq 1 ] && [ -n "${WHEEL_CACHE:-}" ]; then
    aws s3 sync "$dir/" "${WHEEL_CACHE%/}/$key/" --no-progress \
        || echo "[compiled] WARNING: could not upload to ${WHEEL_CACHE%/}/$key/ (need s3:PutObject)" >&2
fi
echo "[compiled] done: $*"

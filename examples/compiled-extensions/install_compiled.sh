#!/usr/bin/env bash
#
# Install Python packages that compile against the instance's PyTorch and CUDA
# (flash-attn, causal-conv1d, mamba-ssm, ...) into ~/venv, with an optional S3
# wheel cache so later instances skip the (often long) build.
#
# Run it on an instance after `aws-bootstrap launch`, from your machine:
#
#   ssh aws-gpu1 'bash -s -- flash-attn==2.8.3' < install_compiled.sh
#   ssh aws-gpu1 'WHEEL_CACHE=s3://my-bucket/wheels bash -s -- flash-attn==2.8.3' < install_compiled.sh
#
# ---------------------------------------------------------------------------
# WHY
# ---------------------------------------------------------------------------
# These packages import torch while they build, so they must be built with
# --no-build-isolation against the torch already in the venv; an isolated build
# installs some other torch (or fails), and a prebuilt wheel from an index rarely
# matches this instance's torch + CUDA. pip's own cache doesn't help a fresh
# instance, hence the S3 cache.
#
# A wheel's filename doesn't record what it was built against, so cached wheels
# live under a key that does:
#
#   torch<version>-cuda<torch's CUDA>-nvcc<toolkit>-cp<python>-<arch>-<libc>-sm<gpu>
#   e.g. torch2.15.0_cu134-cuda13.4-nvcc13.5-cp312-x86_64-glibc2.39-sm89
#
# Any difference there gets its own key, and so a fresh build.
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
# * pip picks the cached wheel, so version specifiers work as usual: a cached
#   wheel that doesn't satisfy the spec is rebuilt. An unpinned name reuses the
#   newest cached version, so pin (==) for reproducibility.
# * URLs and paths (git+https://..., ./src) always build and aren't uploaded:
#   the cache can only be looked up by a package name and version.
# * Build dependencies are not resolved: list packages that need compiling in
#   dependency order (causal-conv1d before mamba-ssm), so each is installed
#   before the next one builds.
# * Each new wheel is uploaded as soon as it is built, so a later failure
#   doesn't lose it. A failed cache read or upload only warns.
# * torch is held at its installed version: a wheel that requires another torch
#   fails to install rather than replacing the CUDA-matched one.

set -euo pipefail

warn() { echo "[compiled] WARNING: $*" >&2; }

is_direct_reference() {
    case "$1" in
        *://* | *" @ "* | .* | /*) return 0 ;;
        *) return 1 ;;
    esac
}

# The cache key (see WHY above), from the venv's python and the GPU.
cache_key() {
    local py=$1 nvcc_version gpu_sm
    nvcc_version=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1)
    gpu_sm=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
    "$py" -c '
import platform, sys, torch
libc, libc_version = platform.libc_ver()
print("-".join([
    "torch" + torch.__version__.replace("+", "_"),
    "cuda" + (torch.version.cuda or "none"),
    "nvcc" + (sys.argv[1] or "none"),
    "cp%d%d" % sys.version_info[:2],
    platform.machine(),
    (libc + libc_version) or "nolibc",
    "sm" + (sys.argv[2] or "none"),
]))' "$nvcc_version" "$gpu_sm"
}

# The cached wheel pip would pick for a spec, or nothing. pip handles the
# specifiers, extras, markers, name normalisation and local versions.
cached_wheel() {
    local py=$1 dir=$2 spec=$3 work=$4
    is_direct_reference "$spec" && return 0
    rm -rf "$work/hit" && mkdir -p "$work/hit"
    # --isolated: only the cache, whatever pip.conf or PIP_* say (the build itself honours them).
    if "$py" -m pip --isolated download --quiet --no-index --no-deps --find-links "$dir" --dest "$work/hit" "$spec" \
        >/dev/null 2>&1; then
        local -a hits=("$work/hit"/*.whl)
        [ -e "${hits[0]}" ] && printf '%s\n' "$dir/$(basename "${hits[0]}")"
    fi
    return 0
}

# Build one wheel for a spec and move it into dest; print its path.
build_wheel() {
    local py=$1 dest=$2 spec=$3 work=$4
    rm -rf "$work/build" && mkdir -p "$work/build"
    # --no-binary: compile here even if the index has a prebuilt wheel (built against some other torch).
    # --no-cache-dir: pip's wheel cache is keyed by the sdist, not by torch, so it could hand back a
    # build made against an earlier torch on this instance.
    "$py" -m pip wheel --no-build-isolation --no-deps --no-cache-dir --no-binary :all: \
        --wheel-dir "$work/build" "$spec" >&2
    local -a built=("$work/build"/*.whl)
    if [ "${#built[@]}" -ne 1 ] || [ ! -e "${built[0]}" ]; then
        echo "[compiled] $spec: expected one wheel from the build, found: ${built[*]}" >&2
        return 1
    fi
    mkdir -p "$dest" && mv -f "${built[0]}" "$dest/"
    printf '%s\n' "$dest/$(basename "${built[0]}")"
}

main() {
    if [ "$#" -eq 0 ]; then
        echo "usage: install_compiled.sh PACKAGE[SPECIFIER] ..." >&2
        exit 2
    fi

    local venv="${VENV:-$HOME/venv}"
    local py="$venv/bin/python"
    # A non-interactive ssh command doesn't read ~/.bashrc, so set up PATH here.
    export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
    if [ ! -x "$py" ]; then
        echo "No venv at $venv (did aws-bootstrap's remote setup run?)" >&2
        exit 1
    fi

    # The tools a non-isolated build expects to find in the venv.
    uv pip install --python "$py" --quiet pip setuptools wheel ninja packaging

    local torch_version key dir cache="" work
    torch_version=$("$py" -c 'import torch; print(torch.__version__)')
    key=$(cache_key "$py")
    dir="$HOME/wheels/$key"
    mkdir -p "$dir"
    if [ -n "${WHEEL_CACHE:-}" ]; then
        cache="${WHEEL_CACHE%/}/$key"
    fi
    work=$(mktemp -d)
    # shellcheck disable=SC2064  # expand $work now: it's local to main
    trap "rm -rf '$work'" EXIT
    echo "torch==$torch_version" >"$work/constraints.txt"
    echo "[compiled] torch $torch_version; cache key $key"

    if [ -n "$cache" ]; then
        aws s3 sync "$cache/" "$dir/" --no-progress || warn "could not read $cache/; building what's missing"
    fi

    local spec wheel
    for spec in "$@"; do
        wheel=$(cached_wheel "$py" "$dir" "$spec" "$work")
        if [ -n "$wheel" ]; then
            echo "[compiled] $spec: using cached $(basename "$wheel")"
        else
            echo "[compiled] $spec: building against torch $torch_version (this can take a while)..."
            if is_direct_reference "$spec"; then
                # Kept out of the cache: it would shadow the index release of the same version.
                wheel=$(build_wheel "$py" "$work/direct" "$spec" "$work")
            else
                wheel=$(build_wheel "$py" "$dir" "$spec" "$work")
                if [ -n "$cache" ]; then
                    aws s3 cp "$wheel" "$cache/" --no-progress || warn "could not upload to $cache/ (needs s3:PutObject)"
                fi
            fi
        fi
        uv pip install --python "$py" --constraint "$work/constraints.txt" "$wheel"
    done
    echo "[compiled] done: $*"
}

# Everything runs from main, so `bash -s < install_compiled.sh` has read the whole
# script before any command that might read stdin starts.
main "$@"

#!/usr/bin/env python3
"""
GPU Throughput Benchmark for AWS EC2 Spot Instances (T4 GPU)

Tests PyTorch GPU utilization with two benchmark modes:
1. CNN on MNIST - lightweight, fast iteration
2. Transformer on synthetic data - more compute-intensive

Reports: iterations/sec, samples/sec, GPU memory usage, and utilization metrics.

Supports multiple precision modes with automatic fallback:
- FP16 AMP (default for Turing/Ampere+)
- FP32 (fallback if AMP fails)
- TF32 (Ampere+ only)
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm


if TYPE_CHECKING:
    from collections.abc import Generator


# -----------------------------------------------------------------------------
# Diagnostic Functions
# -----------------------------------------------------------------------------


def run_cuda_diagnostics(device: torch.device) -> dict[str, bool]:
    """
    Run diagnostic tests to verify CUDA/cuBLAS functionality.
    Returns dict of test_name -> passed.
    """
    results: dict[str, bool] = {}

    if device.type != "cuda":
        print("  Skipping CUDA diagnostics (CPU mode)")
        return results

    print("\n" + "-" * 40)
    print("Running CUDA Diagnostics")
    print("-" * 40)

    # Test 1: Basic FP32 matmul
    try:
        a = torch.randn(256, 256, device=device)
        b = torch.randn(256, 256, device=device)
        _c = torch.mm(a, b)
        torch.cuda.synchronize()
        results["fp32_matmul"] = True
        print("  ✓ FP32 matmul: PASSED")
    except Exception as e:
        results["fp32_matmul"] = False
        print(f"  ✗ FP32 matmul: FAILED - {e}")

    # Test 2: FP16 matmul (no autocast)
    try:
        a = torch.randn(256, 256, device=device, dtype=torch.float16)
        b = torch.randn(256, 256, device=device, dtype=torch.float16)
        _c = torch.mm(a, b)
        torch.cuda.synchronize()
        results["fp16_matmul"] = True
        print("  ✓ FP16 matmul: PASSED")
    except Exception as e:
        results["fp16_matmul"] = False
        print(f"  ✗ FP16 matmul: FAILED - {e}")

    # Test 3: FP16 matmul with autocast
    try:
        a = torch.randn(256, 256, device=device)
        b = torch.randn(256, 256, device=device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            _c = torch.mm(a, b)
        torch.cuda.synchronize()
        results["fp16_autocast"] = True
        print("  ✓ FP16 autocast matmul: PASSED")
    except Exception as e:
        results["fp16_autocast"] = False
        print(f"  ✗ FP16 autocast matmul: FAILED - {e}")

    # Test 4: Linear layer with autocast (common GEMM pattern)
    try:
        linear = nn.Linear(512, 512).to(device)
        x = torch.randn(64, 512, device=device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            _y = linear(x)
        torch.cuda.synchronize()
        results["fp16_linear"] = True
        print("  ✓ FP16 linear layer: PASSED")
    except Exception as e:
        results["fp16_linear"] = False
        print(f"  ✗ FP16 linear layer: FAILED - {e}")

    # Test 5: Conv2d with autocast
    try:
        conv = nn.Conv2d(64, 128, 3, padding=1).to(device)
        x = torch.randn(16, 64, 32, 32, device=device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            _y = conv(x)
        torch.cuda.synchronize()
        results["fp16_conv2d"] = True
        print("  ✓ FP16 conv2d: PASSED")
    except Exception as e:
        results["fp16_conv2d"] = False
        print(f"  ✗ FP16 conv2d: FAILED - {e}")

    # Test 6: Batched matmul (transformer attention pattern)
    try:
        # Simulates attention: (batch, heads, seq, dim) @ (batch, heads, dim, seq)
        a = torch.randn(8, 8, 128, 64, device=device, dtype=torch.float16)
        b = torch.randn(8, 8, 64, 128, device=device, dtype=torch.float16)
        _c = torch.matmul(a, b)
        torch.cuda.synchronize()
        results["fp16_batched_matmul"] = True
        print("  ✓ FP16 batched matmul: PASSED")
    except Exception as e:
        results["fp16_batched_matmul"] = False
        print(f"  ✗ FP16 batched matmul: FAILED - {e}")

    print("-" * 40)

    # Summary
    passed = sum(results.values())
    total = len(results)
    print(f"Diagnostics: {passed}/{total} tests passed")

    if passed < total:
        failed_tests = [k for k, v in results.items() if not v]
        print(f"Failed tests: {', '.join(failed_tests)}")
        print("\nRecommendation: Use --precision fp32 to bypass FP16 issues")

    print("-" * 40 + "\n")
    return results


class PrecisionMode(Enum):
    """Supported precision modes for training."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    TF32 = "tf32"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    batch_size: int = 256
    num_warmup_batches: int = 10
    num_benchmark_batches: int = 100
    num_workers: int = 4
    pin_memory: bool = True
    precision: PrecisionMode = PrecisionMode.FP16


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    model_name: str
    total_samples: int
    total_time_sec: float
    peak_memory_mb: float
    avg_batch_time_ms: float
    precision_mode: str

    @property
    def samples_per_sec(self) -> float:
        return self.total_samples / self.total_time_sec

    @property
    def batches_per_sec(self) -> float:
        return 1000.0 / self.avg_batch_time_ms

    def __str__(self) -> str:
        return (
            f"\n{'=' * 60}\n"
            f"Benchmark Results: {self.model_name}\n"
            f"{'=' * 60}\n"
            f"  Precision mode: {self.precision_mode}\n"
            f"  Total samples processed: {self.total_samples:,}\n"
            f"  Total time: {self.total_time_sec:.2f}s\n"
            f"  Throughput: {self.samples_per_sec:,.1f} samples/sec\n"
            f"  Throughput: {self.batches_per_sec:.1f} batches/sec\n"
            f"  Avg batch time: {self.avg_batch_time_ms:.2f}ms\n"
            f"  Peak GPU memory: {self.peak_memory_mb:.1f}MB\n"
            f"{'=' * 60}\n"
        )


# -----------------------------------------------------------------------------
# CNN Model for MNIST
# -----------------------------------------------------------------------------


class MNISTConvNet(nn.Module):
    """
    Simple but non-trivial CNN for MNIST.
    ~1.2M parameters - enough to stress GPU without being excessive.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# -----------------------------------------------------------------------------
# Transformer Model (GPT-style decoder)
# -----------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Single transformer decoder block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm architecture
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    Small GPT-style transformer for benchmarking.
    ~25M parameters - representative of real workloads.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} > max {self.max_seq_len}"

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        x = self.dropout(self.token_emb(x) + self.pos_emb(positions))

        # Causal mask for autoregressive modeling
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        x = self.ln_f(x)
        return self.head(x)


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------


def get_mnist_loader(config: BenchmarkConfig, device: torch.device) -> DataLoader:
    """Load MNIST dataset with standard preprocessing."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = datasets.MNIST(root="/tmp/data", train=True, download=True, transform=transform)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )


def get_synthetic_text_loader(
    config: BenchmarkConfig,
    vocab_size: int = 32000,
    seq_len: int = 256,
    num_samples: int = 50000,
) -> DataLoader:
    """Generate synthetic token sequences for transformer benchmarking."""
    # Random token IDs (simulates real tokenized text distribution)
    data = torch.randint(0, vocab_size, (num_samples, seq_len))
    # Labels are next-token shifted (standard LM objective)
    labels = torch.randint(0, vocab_size, (num_samples, seq_len))

    dataset = TensorDataset(data, labels)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Synthetic data is fast enough
        pin_memory=True,
    )


# -----------------------------------------------------------------------------
# Benchmark Runner
# -----------------------------------------------------------------------------


@contextmanager
def cuda_timer(device: torch.device) -> Generator[dict[str, float]]:
    """Context manager for accurate CUDA timing using events."""
    result: dict[str, float] = {}

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        yield result
        end_event.record()
        torch.cuda.synchronize(device)
        result["elapsed_ms"] = start_event.elapsed_time(end_event)
    else:
        start = time.perf_counter()
        yield result
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000


def run_benchmark(
    model: nn.Module,
    loader: DataLoader,
    config: BenchmarkConfig,
    device: torch.device,
    model_name: str,
    precision: PrecisionMode,
    is_lm: bool = False,
) -> BenchmarkResult:
    """
    Run training benchmark with warmup phase.

    Args:
        model: PyTorch model to benchmark
        loader: DataLoader providing batches
        config: Benchmark configuration
        device: Target device
        model_name: Name for reporting
        precision: Precision mode to use
        is_lm: If True, use language modeling loss (ignore_index=-100)
    """
    model = model.to(device)
    model.train()

    # Configure precision-specific settings
    use_amp = precision in (PrecisionMode.FP16, PrecisionMode.BF16)
    amp_dtype = torch.float16 if precision == PrecisionMode.FP16 else torch.bfloat16

    # GradScaler is only needed for FP16 (BF16 has sufficient dynamic range)
    use_scaler = precision == PrecisionMode.FP16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data_iter = iter(loader)
    batch_times: list[float] = []

    total_batches = config.num_warmup_batches + config.num_benchmark_batches

    print(f"\nRunning {model_name} benchmark...")
    print(f"  Precision: {precision.value}")
    print(f"  AMP enabled: {use_amp}")
    print(f"  GradScaler enabled: {use_scaler}")
    print(f"  Warmup batches: {config.num_warmup_batches}")
    print(f"  Benchmark batches: {config.num_benchmark_batches}")
    print(f"  Batch size: {config.batch_size}")

    pbar = tqdm(range(total_batches), desc=model_name, unit="batch")
    for batch_idx in pbar:
        # Get next batch, cycling if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        inputs = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)

        with cuda_timer(device) as timer:
            optimizer.zero_grad(set_to_none=True)

            # Use autocast only when AMP is enabled
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                outputs = model(inputs)

                if is_lm:
                    # Reshape for cross-entropy: (batch * seq_len, vocab_size)
                    loss = F.cross_entropy(
                        outputs.view(-1, outputs.size(-1)),
                        targets.view(-1),
                    )
                else:
                    loss = F.cross_entropy(outputs, targets)

            # Backward pass with optional gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Only record times after warmup
        is_benchmark = batch_idx >= config.num_warmup_batches
        if is_benchmark:
            batch_times.append(timer["elapsed_ms"])

        # Update progress bar
        phase = "bench" if is_benchmark else "warmup"
        postfix: dict[str, str] = {"phase": phase, "loss": f"{loss.item():.4f}"}
        if is_benchmark and batch_times:
            sps = config.batch_size / (batch_times[-1] / 1000)
            postfix["samples/s"] = f"{sps:,.0f}"
        if device.type == "cuda":
            mem_mb = torch.cuda.memory_allocated(device) / (1024**2)
            postfix["gpu_mem"] = f"{mem_mb:.0f}MB"
        pbar.set_postfix(postfix)

    # Compute statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    total_time = sum(batch_times) / 1000  # Convert to seconds
    total_samples = config.num_benchmark_batches * config.batch_size

    peak_memory = 0.0
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return BenchmarkResult(
        model_name=model_name,
        total_samples=total_samples,
        total_time_sec=total_time,
        peak_memory_mb=peak_memory,
        avg_batch_time_ms=avg_batch_time,
        precision_mode=precision.value,
    )


# -----------------------------------------------------------------------------
# System Information and GPU Configuration
# -----------------------------------------------------------------------------


def get_gpu_architecture(device: torch.device) -> tuple[int, int]:
    """Get GPU compute capability (major, minor)."""
    if device.type != "cuda":
        return (0, 0)
    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor)


def configure_precision(device: torch.device, requested: PrecisionMode) -> PrecisionMode:
    """
    Configure and validate precision mode based on GPU capabilities.

    GPU Architecture Reference:
    - Turing (T4): sm_75 - Supports FP16 tensor cores, NO native BF16, NO TF32
    - Ampere (A100, A10, 3090): sm_80/86 - Supports FP16, BF16, TF32
    - Hopper (H100): sm_90 - Full support for all modes

    Returns the actual precision mode that will be used.
    """
    if device.type != "cuda":
        print("  CPU mode: Using FP32")
        return PrecisionMode.FP32

    major, minor = get_gpu_architecture(device)
    sm_version = major * 10 + minor

    print(f"  GPU compute capability: sm_{sm_version}")

    # =========================================================================
    # CRITICAL: Disable problematic cuBLAS features that can cause GEMM errors
    # These settings improve stability on older architectures like Turing (T4)
    # =========================================================================

    # Disable TF32 on non-Ampere hardware (TF32 is Ampere+ only)
    if sm_version < 80:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("  TF32 disabled (requires sm_80+)")

    # Disable reduced precision reductions in FP16 GEMMs
    # This can cause overflow/execution failures on some cuBLAS versions
    # See: https://docs.pytorch.org/docs/stable/notes/cuda.html
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    print("  FP16 reduced precision reduction disabled (for stability)")

    # Also disable BF16 reduced precision reduction for consistency
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    print("  BF16 reduced precision reduction disabled (for stability)")

    # TF32 requires Ampere or newer (sm_80+)
    if requested == PrecisionMode.TF32:
        if sm_version >= 80:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  TF32 mode enabled for matmul and cuDNN")
            return PrecisionMode.TF32
        else:
            print(f"  WARNING: TF32 requires sm_80+, but GPU is sm_{sm_version}")
            print("  Falling back to FP16 AMP")
            requested = PrecisionMode.FP16

    # BF16 requires Ampere or newer (sm_80+) for efficient operation
    if requested == PrecisionMode.BF16:
        if sm_version >= 80 and torch.cuda.is_bf16_supported():
            print("  BF16 mode enabled")
            return PrecisionMode.BF16
        else:
            print(f"  WARNING: BF16 not efficiently supported on sm_{sm_version}")
            print("  Falling back to FP16 AMP")
            requested = PrecisionMode.FP16

    # FP16 works on Volta (sm_70) and newer
    if requested == PrecisionMode.FP16:
        if sm_version >= 70:
            print("  FP16 AMP mode enabled")
            return PrecisionMode.FP16
        else:
            print(f"  WARNING: FP16 tensor cores require sm_70+, but GPU is sm_{sm_version}")
            print("  Falling back to FP32")
            return PrecisionMode.FP32

    # FP32 always works
    print("  FP32 mode (no mixed precision)")
    return PrecisionMode.FP32


def print_system_info(requested_precision: PrecisionMode) -> tuple[torch.device, PrecisionMode]:
    """Print system and CUDA information, return device and actual precision mode."""
    print("\n" + "=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version.split()[0]}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")

        cudnn_version = torch.backends.cudnn.version()
        if cudnn_version:
            print(f"cuDNN version: {cudnn_version}")

        print(f"Device count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total memory: {props.total_memory / (1024**3):.1f}GB")
            print(f"  SM count: {props.multi_processor_count}")

        print("\nPrecision Configuration:")
        actual_precision = configure_precision(device, requested_precision)

        # Set deterministic cuBLAS workspace config for stability
        # This can help avoid sporadic GEMM failures
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            print("  Set CUBLAS_WORKSPACE_CONFIG=:4096:8 for stability")

    else:
        device = torch.device("cpu")
        actual_precision = PrecisionMode.FP32
        print("CUDA available: No (running on CPU)")
        print("WARNING: GPU benchmark results will not be representative!")

    print("=" * 60)
    return device, actual_precision


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU Throughput Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["cnn", "transformer", "both"],
        default="both",
        help="Benchmark mode: cnn (MNIST), transformer (synthetic LM), or both",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for CNN training",
    )
    parser.add_argument(
        "--transformer-batch-size",
        type=int,
        default=32,
        help="Batch size for transformer training (smaller due to large vocab logits)",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=10,
        help="Number of warmup batches (not timed)",
    )
    parser.add_argument(
        "--benchmark-batches",
        type=int,
        default=100,
        help="Number of batches to benchmark",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16", "tf32"],
        default="fp16",
        help="Precision mode: fp32 (full), fp16 (AMP), bf16 (AMP), tf32 (Ampere+)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run CUDA/cuBLAS diagnostic tests before benchmarking",
    )
    args = parser.parse_args()

    requested_precision = PrecisionMode(args.precision)
    device, actual_precision = print_system_info(requested_precision)

    # Run diagnostics if requested
    if args.diagnose:
        diag_results = run_cuda_diagnostics(device)
        # If FP16 tests fail, suggest using FP32
        fp16_tests = ["fp16_matmul", "fp16_autocast", "fp16_linear", "fp16_batched_matmul"]
        fp16_failures = [t for t in fp16_tests if t in diag_results and not diag_results[t]]
        if fp16_failures and actual_precision == PrecisionMode.FP16:
            print("WARNING: FP16 diagnostic tests failed. Switching to FP32.")
            actual_precision = PrecisionMode.FP32

    config = BenchmarkConfig(
        batch_size=args.batch_size,
        num_warmup_batches=args.warmup_batches,
        num_benchmark_batches=args.benchmark_batches,
        precision=actual_precision,
    )

    results: list[BenchmarkResult] = []

    # CNN Benchmark
    if args.mode in ("cnn", "both"):
        model = MNISTConvNet()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\nMNIST CNN parameters: {param_count:,}")

        loader = get_mnist_loader(config, device)

        try:
            result = run_benchmark(model, loader, config, device, "MNIST CNN", actual_precision, is_lm=False)
            results.append(result)
            print(result)
        except RuntimeError as e:
            if "CUBLAS" in str(e) or "cuBLAS" in str(e):
                print(f"\n*** cuBLAS error encountered with {actual_precision.value} ***")
                print(f"Error: {e}")
                print("\nRetrying with FP32 (no AMP)...")

                # Cleanup and retry with FP32
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                model = MNISTConvNet()
                result = run_benchmark(model, loader, config, device, "MNIST CNN", PrecisionMode.FP32, is_lm=False)
                results.append(result)
                print(result)
            else:
                raise

        # Cleanup
        del model, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Transformer Benchmark
    if args.mode in ("transformer", "both"):
        transformer_config = BenchmarkConfig(
            batch_size=args.transformer_batch_size,
            num_warmup_batches=args.warmup_batches,
            num_benchmark_batches=args.benchmark_batches,
            precision=actual_precision,
        )

        model = MiniGPT()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\nMiniGPT parameters: {param_count:,}")

        loader = get_synthetic_text_loader(transformer_config)

        try:
            result = run_benchmark(
                model, loader, transformer_config, device, "MiniGPT Transformer", actual_precision, is_lm=True
            )
            results.append(result)
            print(result)
        except RuntimeError as e:
            if "CUBLAS" in str(e) or "cuBLAS" in str(e):
                print(f"\n*** cuBLAS error encountered with {actual_precision.value} ***")
                print(f"Error: {e}")
                print("\nRetrying with FP32 (no AMP)...")

                # Cleanup and retry with FP32
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                model = MiniGPT()
                result = run_benchmark(
                    model, loader, transformer_config, device, "MiniGPT Transformer", PrecisionMode.FP32, is_lm=True
                )
                results.append(result)
                print(result)
            else:
                raise

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        for r in results:
            print(f"{r.model_name} ({r.precision_mode}):")
            print(f"  {r.samples_per_sec:,.1f} samples/sec | {r.peak_memory_mb:.0f}MB peak")
        print("=" * 60)


if __name__ == "__main__":
    main()

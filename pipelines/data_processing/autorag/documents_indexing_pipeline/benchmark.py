#!/usr/bin/env python3
"""Benchmark runner for the AutoRAG documents indexing pipeline.

Executes each pipeline component locally by calling the component's Python
function directly (no KFP cluster required), and reports wall-clock time and
peak resident-set-size for each step.

Usage:
    python -m kfp_components.pipelines.data_processing.autorag.documents_indexing_pipeline.benchmark \\
        --bucket my-bucket --prefix docs/

    # With indexing (requires a running Llama Stack server):
    python -m kfp_components.pipelines.data_processing.autorag.documents_indexing_pipeline.benchmark \\
        --bucket my-bucket --prefix docs/ \\
        --embedding-model-id my-model-id

Required environment variables (S3 access):
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_S3_ENDPOINT
    AWS_DEFAULT_REGION   (optional)

Optional environment variables (documents_indexing step):
    LLAMA_STACK_CLIENT_BASE_URL
    LLAMA_STACK_CLIENT_API_KEY

The documents_indexing step is skipped automatically when
LLAMA_STACK_CLIENT_BASE_URL or --embedding-model-id is absent, or when
--skip-indexing is passed.

main() runs several sampling_max_size values in a loop, writes one JSON file per
iteration under benchmark_run_json/, and continues after failures (see
iteration_logical_success and iteration_exception in each file).
"""

import json
import os
import resource
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from kfp_components.components.data_processing.autorag.documents_discovery.component import (
    documents_discovery,
)
from kfp_components.components.data_processing.autorag.documents_indexing.component import (
    documents_indexing,
)
from kfp_components.components.data_processing.autorag.text_extraction.component import (
    text_extraction,
)


# ---------------------------------------------------------------------------
# Artifact stub
# ---------------------------------------------------------------------------


class _LocalArtifact:
    """Minimal stub satisfying the .path interface expected by KFP components."""

    def __init__(self, path: Path) -> None:
        self.path = str(path)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Timing and status for a single pipeline step."""

    name: str
    skipped: bool
    success: bool
    duration_s: float = 0.0
    peak_rss_mb: float = 0.0
    error: Optional[str] = None
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_python_func(component):
    """Return the raw Python function from a KFP @dsl.component object."""
    fn = getattr(component, "python_func", None)
    if fn is None:
        raise TypeError(
            f"Cannot extract python_func from {type(component).__name__}. "
            "Ensure kfp >= 2.0 is installed and the package is importable."
        )
    return fn


def _rss_mb() -> float:
    """Current process peak resident set size in MiB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on Linux, kibibytes on macOS
    if sys.platform == "darwin":
        return usage.ru_maxrss / 1024 / 1024
    return usage.ru_maxrss / 1024


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def _check_s3_env() -> list[str]:
    """Return a list of missing required S3 environment variable names."""
    required = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT"]
    return [k for k in required if not os.environ.get(k)]


def _step_results_to_jsonable(results: list[StepResult]) -> list[dict[str, Any]]:
    """Convert StepResult list to JSON-serializable dicts."""
    return [
        {
            "name": r.name,
            "skipped": r.skipped,
            "success": r.success,
            "duration_s": r.duration_s,
            "peak_rss_mb": r.peak_rss_mb,
            "error": r.error,
            "info": r.info,
        }
        for r in results
    ]


def _save_benchmark_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _sampling_size_slug(gib: float) -> str:
    """Safe filename fragment for a sampling_max_size value in GiB."""
    return f"{gib:g}".replace(".", "p")


# ---------------------------------------------------------------------------
# Per-step runners
# ---------------------------------------------------------------------------


def _step_discovery(
    work_dir: Path,
    bucket: str,
    prefix: str,
    sampling_enabled: bool,
    sampling_max_size: float,
) -> StepResult:
    artifact_dir = work_dir / "discovered_documents"
    artifact_dir.mkdir(parents=True)

    rss_before = _rss_mb()
    t0 = time.perf_counter()
    try:
        _get_python_func(documents_discovery)(
            input_data_bucket_name=bucket,
            input_data_path=prefix,
            test_data=None,
            sampling_enabled=sampling_enabled,
            sampling_max_size=sampling_max_size,
            discovered_documents=_LocalArtifact(artifact_dir),
        )
        duration = time.perf_counter() - t0

        descriptor_path = artifact_dir / "documents_descriptor.json"
        with open(descriptor_path) as f:
            desc = json.load(f)

        return StepResult(
            name="documents_discovery",
            skipped=False,
            success=True,
            duration_s=duration,
            peak_rss_mb=max(_rss_mb() - rss_before, 0.0),
            info={
                "documents": desc.get("count", 0),
                "total_size_mb": round(desc.get("total_size_bytes", 0) / 1024**2, 1),
            },
        )
    except Exception as exc:
        return StepResult(
            name="documents_discovery",
            skipped=False,
            success=False,
            duration_s=time.perf_counter() - t0,
            error=str(exc),
        )


def _step_extraction(work_dir: Path) -> StepResult:
    discovered_dir = work_dir / "discovered_documents"
    extracted_dir = work_dir / "extracted_text"
    extracted_dir.mkdir(parents=True)

    rss_before = _rss_mb()
    t0 = time.perf_counter()
    try:
        _get_python_func(text_extraction)(
            documents_descriptor=_LocalArtifact(discovered_dir),
            extracted_text=_LocalArtifact(extracted_dir),
        )
        duration = time.perf_counter() - t0

        md_files = list(extracted_dir.glob("*.md"))
        return StepResult(
            name="text_extraction",
            skipped=False,
            success=True,
            duration_s=duration,
            peak_rss_mb=max(_rss_mb() - rss_before, 0.0),
            info={"extracted_files": len(md_files)},
        )
    except Exception as exc:
        return StepResult(
            name="text_extraction",
            skipped=False,
            success=False,
            duration_s=time.perf_counter() - t0,
            error=str(exc),
        )


def _step_indexing(
    work_dir: Path,
    embedding_model_id: str,
    llama_stack_vector_database_id: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
) -> StepResult:
    extracted_dir = work_dir / "extracted_text"

    rss_before = _rss_mb()
    t0 = time.perf_counter()
    try:
        _get_python_func(documents_indexing)(
            embedding_model_id=embedding_model_id,
            extracted_text=_LocalArtifact(extracted_dir),
            llama_stack_vector_database_id=llama_stack_vector_database_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
        )
        duration = time.perf_counter() - t0
        return StepResult(
            name="documents_indexing",
            skipped=False,
            success=True,
            duration_s=duration,
            peak_rss_mb=max(_rss_mb() - rss_before, 0.0),
        )
    except Exception as exc:
        return StepResult(
            name="documents_indexing",
            skipped=False,
            success=False,
            duration_s=time.perf_counter() - t0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run(
    bucket: str,
    prefix: str,
    *,
    work_dir: Optional[Path] = None,
    sampling_enabled: bool = True,
    sampling_max_size: float = 1.0,
    skip_indexing: bool = False,
    embedding_model_id: Optional[str] = None,
    llama_stack_vector_database_id: str = "ls_milvus",
    chunk_size: int = 1024,
    chunk_overlap: int = 0,
    batch_size: int = 20,
) -> list[StepResult]:
    """Run the full pipeline locally and return per-step benchmark results.

    Args:
        bucket: S3 bucket name containing the source documents.
        prefix: S3 key prefix (folder path) within the bucket.
        work_dir: Directory for intermediate artifacts. A temporary directory is
            created and cleaned up automatically when this is None.
        sampling_enabled: Whether documents_discovery should sample documents.
        sampling_max_size: Maximum total size of sampled documents in GiB.
        skip_indexing: Force-skip the documents_indexing step.
        embedding_model_id: Embedding model ID required by documents_indexing.
            Leaving this None skips that step.
        llama_stack_vector_database_id: Llama Stack vector database provider ID.
        chunk_size: Chunk size in characters for text splitting.
        chunk_overlap: Chunk overlap in characters.
        batch_size: Documents per indexing batch (0 = all at once).

    Returns:
        List of StepResult objects, one per pipeline step.
    """
    managed = work_dir is None
    if managed:
        work_dir = Path(tempfile.mkdtemp(prefix="autorag_bench_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)


    results: list[StepResult] = []

    # ── Step 1: documents_discovery ────────────────────────────────────
    print(f"[1/3] Running documents_discovery  (bucket={bucket!r}, prefix={prefix!r})")
    result = _step_discovery(work_dir, bucket, prefix, sampling_enabled, sampling_max_size)
    results.append(result)
    if not result.success:
        print(f"      FAILED: {result.error}")
        return results
    print(f"      OK  ({_fmt_duration(result.duration_s)}, {result.info})")

    # ── Step 2: text_extraction ────────────────────────────────────────
    print("[2/3] Running text_extraction")
    result = _step_extraction(work_dir)
    results.append(result)
    if not result.success:
        print(f"      FAILED: {result.error}")
        return results
    print(f"      OK  ({_fmt_duration(result.duration_s)}, {result.info})")

    # ── Step 3: documents_indexing (optional) ──────────────────────────
    llama_url = os.environ.get("LLAMA_STACK_CLIENT_BASE_URL")
    if skip_indexing:
        reason = "--skip-indexing flag"
    elif not llama_url:
        reason = "LLAMA_STACK_CLIENT_BASE_URL not set"
    elif not embedding_model_id:
        reason = "--embedding-model-id not provided"
    else:
        reason = None

    if reason:
        print(f"[3/3] Skipping documents_indexing  ({reason})")
        results.append(StepResult(name="documents_indexing", skipped=True, success=True))
    else:
        print(f"[3/3] Running documents_indexing  (model={embedding_model_id!r})")
        result = _step_indexing(
            work_dir,
            embedding_model_id,
            llama_stack_vector_database_id,
            chunk_size,
            chunk_overlap,
            batch_size,
        )
        results.append(result)
        if not result.success:
            print(f"      FAILED: {result.error}")
        else:
            print(f"      OK  ({_fmt_duration(result.duration_s)})")

    return results


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def print_report(results: list[StepResult]) -> None:
    """Print a formatted timing table to stdout."""
    W_NAME = 28
    W_STATUS = 10
    W_DUR = 12
    W_RSS = 12
    W_INFO = 0  # expands freely

    header = (
        f"{'Component':<{W_NAME}}"
        f"{'Status':<{W_STATUS}}"
        f"{'Duration':>{W_DUR}}"
        f"{'Peak RSS':>{W_RSS}}"
        f"  Info"
    )
    rule = "─" * 80

    active = [r for r in results if not r.skipped]
    total_s = sum(r.duration_s for r in active)
    all_ok = all(r.success for r in active)

    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + "  AutoRAG Documents Indexing Pipeline — Benchmark Results".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print(header)
    print(rule)

    for r in results:
        if r.skipped:
            status, dur_str, rss_str = "SKIPPED", "—", "—"
        elif r.success:
            status = "OK"
            dur_str = _fmt_duration(r.duration_s)
            rss_str = f"{r.peak_rss_mb:.0f} MiB" if r.peak_rss_mb > 0 else "—"
        else:
            status = "FAILED"
            dur_str = _fmt_duration(r.duration_s)
            rss_str = f"{r.peak_rss_mb:.0f} MiB" if r.peak_rss_mb > 0 else "—"

        info_parts = [f"{k}={v}" for k, v in r.info.items()]
        if r.error:
            # truncate long errors so the table stays readable
            info_parts.append(f"error={r.error[:60]}{'…' if len(r.error) > 60 else ''}")
        info_str = ", ".join(info_parts)

        print(
            f"{r.name:<{W_NAME}}"
            f"{status:<{W_STATUS}}"
            f"{dur_str:>{W_DUR}}"
            f"{rss_str:>{W_RSS}}"
            f"  {info_str}"
        )

    print(rule)
    total_status = "OK" if all_ok else "FAILED (see above)"
    print(
        f"{'TOTAL':<{W_NAME}}"
        f"{total_status:<{W_STATUS}}"
        f"{_fmt_duration(total_s):>{W_DUR}}"
    )
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""

    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["AWS_DEFAULT_REGION"] = "us-south"
    os.environ["AWS_S3_ENDPOINT"] = "https://minio-api-redhat-ods-operator.apps.rosa.ai-eng-gpu.socc.p3.openshiftapps.com"
    os.environ["LLAMA_STACK_CLIENT_API_KEY"] = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJHVGUzTWtHT0U3NTJRT2hYRHZXRm84amhTbVFiaExSZGFqSjBJN2VKcjdVIn0.eyJleHAiOjE3Nzk4ODcyNDIsImlhdCI6MTc3MjExMTI0MiwianRpIjoib25ydHJvOmI2ZGU1ZGYwLWQ5NWItOTc5ZC1jYjNmLWMxMzA2ZTA2ZTZjMyIsImlzcyI6Imh0dHBzOi8va2V5Y2xvYWstaW5ncmVzcy1rZXljbG9hay5hcHBzLnJvc2EuYWktZW5nLWdwdS5zb2NjLnAzLm9wZW5zaGlmdGFwcHMuY29tL3JlYWxtcy9sbGFtYXN0YWNrLWRlbW8iLCJhdWQiOiJhY2NvdW50Iiwic3ViIjoiYzlkM2ZlM2MtZTc5ZC00MWJmLTgzNDMtMzNiY2U2MzhmMWU4IiwidHlwIjoiQmVhcmVyIiwiYXpwIjoibGxhbWFzdGFjayIsInNpZCI6ImE4MjY1ZmI1LTllYWUtZDEwYi1jZjVlLTQ1Mzk3ODUzNTYyMCIsImFjciI6IjEiLCJhbGxvd2VkLW9yaWdpbnMiOlsiLyoiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbImRlZmF1bHQtcm9sZXMtbGxhbWFzdGFjay1kZW1vIiwib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoicHJvZmlsZSBlbWFpbCIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYW1lIjoiVXNlciBUaGUgRmlyc3QiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJ1c2VyMSIsImdpdmVuX25hbWUiOiJVc2VyIiwiZmFtaWx5X25hbWUiOiJUaGUgRmlyc3QiLCJlbWFpbCI6InVzZXIxQGtleWNsb2FrLm9yZyJ9.ASVc7X63o0LldMFf8POoowDqI-aGB1biUhoBk_RxFj3O4VptBy2HzsoGPVou8CpByZrbTkC0SPOR3FtkPB4mic3G2XAOlWbg7WlG9-4GBA6-qxzpOj4-QWHGrIDhU2vaaieeWQUg_ylywgjm9n3-ZaNE1RqJHpbB2Y4n84nqfXWWhTG9tQKwpM1g-14Wy2W-gAy6slzeeWdnzmfOC2yXzrvMXoqctVWJWRFxLpG8effyoHayjuqbYyGYYxqOFD5EFl7FuurYo3BIECFlB_IgUXg6liZUSkD76q-zoYUr5gZ0KuWcNeXw_zlpG8PVP3GabgNXLB_h7f-UI6GPs3wxzA"
    os.environ["LLAMA_STACK_CLIENT_BASE_URL"] = "https://llama-stack-secure-redhat-ods-operator.apps.rosa.ai-eng-gpu.socc.p3.openshiftapps.com"

    missing = _check_s3_env()
    if missing:
        print(f"Missing S3 environment variables: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    bucket = "autorag-datasets"
    prefix = "math-docs-dataset"
    sampling_enabled = True
    work_dir = ""
    sampling_max_sizes_gib = [0.01, 0.3, 0.5, 0.7, 1.0, 1.0, 0.5, 0.2, 0.3, 0.5, 1.0]
    json_output_dir = Path("benchmark_run_json_4")
    skip_indexing = False
    embedding_model_id = "vllm-embedding/bge-m3"
    llama_stack_vector_database_id = "ls_milvus"
    chunk_size = 1024
    chunk_overlap = 128
    batch_size = 10

    print(f"\nBucket  : {bucket}")
    print(f"Prefix  : {prefix!r}")
    print(f"Sampling: enabled={sampling_enabled}, max_sizes (GiB)={sampling_max_sizes_gib}")
    print(f"JSON out: {json_output_dir.resolve()}")
    if work_dir:
        print(f"Work dir: {work_dir}")

    run_started = datetime.now(timezone.utc).isoformat()
    any_iteration_failed = False

    for idx, sampling_max_size in enumerate(sampling_max_sizes_gib):
        slug = _sampling_size_slug(sampling_max_size)
        json_path = json_output_dir / f"benchmark_sampling_max_{slug}_gib.json"
        print(f"\n{'=' * 60}")
        print(f"Iteration {idx + 1}/{len(sampling_max_sizes_gib)}  sampling_max_size={sampling_max_size} GiB")
        print(f"{'=' * 60}")

        payload: dict[str, Any] = {
            "schema": "autorag_documents_indexing_benchmark_iteration",
            "run_started_utc": run_started,
            "iteration_index": idx,
            "bucket": bucket,
            "prefix": prefix,
            "sampling_enabled": sampling_enabled,
            "sampling_max_size_gib": sampling_max_size,
            "json_path": str(json_path),
        }

        # try:
        results = run(
            bucket=bucket,
            prefix=prefix,
            work_dir=Path(work_dir) if work_dir else None,
            sampling_enabled=sampling_enabled,
            sampling_max_size=sampling_max_size,
            skip_indexing=skip_indexing,
            embedding_model_id=embedding_model_id,
            llama_stack_vector_database_id=llama_stack_vector_database_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
        )
        payload["iteration_exception"] = None
        payload["steps"] = _step_results_to_jsonable(results)
        step_failed = [r for r in results if not r.skipped and not r.success]
        payload["iteration_logical_success"] = len(step_failed) == 0
        if step_failed:
            any_iteration_failed = True
        print_report(results)
        # except Exception as exc:
        #     payload["iteration_exception"] = repr(exc)
        #     payload["steps"] = None
        #     payload["iteration_logical_success"] = False
        #     any_iteration_failed = True
        #     print(f"      ITERATION CRASHED: {exc!r}", file=sys.stderr)

        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _save_benchmark_json(json_path, payload)
        print(f"Wrote {json_path}")

    sys.exit(1 if any_iteration_failed else 0)


if __name__ == "__main__":
    main()

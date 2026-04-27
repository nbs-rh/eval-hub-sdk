"""EvalCard and Environment Card metadata models.

EvalCard implements the standardized evaluation disclosure format of
Dhar et al. (arXiv:2511.21695). Environment Card is a novel specification
capturing the complete operational context of an evaluation run.
"""

from __future__ import annotations

import logging
import platform
import sys
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EvalCard — Dhar et al. (arXiv:2511.21695)
# ---------------------------------------------------------------------------


class CapabilityEvalEntry(BaseModel):
    """One row in the EvalCard Capability Evaluations table."""

    ability: str = Field(
        description="Capability category (e.g. knowledge, reasoning, math, coding)"
    )
    benchmark: str = Field(description="Benchmark name and version")
    metric: str = Field(
        description="Evaluation metric (e.g. exact_match, f1, accuracy)"
    )
    zero_shot: float | str | None = Field(
        default=None, description="Score under zero-shot prompting"
    )
    alt_prompting: float | str | None = Field(
        default=None, description="Score under alternative prompting"
    )
    alt_prompting_description: str | None = Field(
        default=None,
        description="Alternative prompting strategy description (e.g. '5-Shot CoT')",
    )


class SafetyEvalEntry(BaseModel):
    """One row in the EvalCard Safety Evaluations table."""

    feature: str = Field(
        description="Safety feature evaluated (e.g. toxicity, bias, jailbreak)"
    )
    benchmark: str = Field(description="Safety benchmark name")
    metric: str = Field(description="Safety metric")
    zero_shot: float | str | None = Field(
        default=None, description="Score under zero-shot prompting"
    )
    alt_prompting: float | str | None = Field(
        default=None, description="Score under alternative prompting"
    )
    alt_prompting_description: str | None = Field(
        default=None, description="Alternative prompting strategy"
    )


class EvalCardMetadata(BaseModel):
    """Standardized evaluation disclosure card (EvalCard).

    Implements Dhar et al. (arXiv:2511.21695): concise, structured evaluation
    summaries that are easy to write, easy to understand, and hard to miss.

    Serialized into ``artifacts["evalhub.eval_card"]`` on ``report_results()``.
    """

    modalities_input: list[str] = Field(
        default_factory=list, description="Input modalities evaluated"
    )
    modalities_output: list[str] = Field(
        default_factory=list, description="Output modalities evaluated"
    )
    languages_count: int | None = Field(
        default=None, description="Number of languages evaluated"
    )
    languages: list[str] = Field(
        default_factory=list, description="ISO 639 language codes evaluated"
    )
    capability_evaluations: list[CapabilityEvalEntry] = Field(
        default_factory=list, description="Capability evaluations table"
    )
    safety_evaluations: list[SafetyEvalEntry] = Field(
        default_factory=list, description="Safety evaluations table"
    )
    developer_footnotes: str | None = Field(
        default=None, description="Free-text evaluation context and caveats"
    )


# ---------------------------------------------------------------------------
# Environment Card — novel specification (this project)
# Five layers: hardware, software, kubernetes, model_identity, run_provenance
# ---------------------------------------------------------------------------

# Total spec fields used for completeness scoring
_ENV_CARD_SPEC_FIELD_COUNT = 26


class EnvironmentCardMetadata(BaseModel):
    """Complete operational context of an evaluation run.

    Captures everything NOT covered by the EvalCard: where the evaluation ran,
    which model was evaluated, and the full provenance chain.

    Serialized into ``artifacts["evalhub.env_card"]`` on ``report_results()``.
    """

    # --- Layer 1: Hardware ---
    gpu_model: str | None = Field(default=None, description="GPU model name")
    gpu_count: int | None = Field(default=None, description="Number of GPUs available")
    gpu_driver_version: str | None = Field(
        default=None, description="GPU driver version"
    )
    cpu_model: str | None = Field(default=None, description="CPU model name")
    total_memory_gb: float | None = Field(
        default=None,
        description="Total system memory in GB (not auto-captured; set by adapter)",
    )
    nvlink_topology: str | None = Field(
        default=None,
        description="NVLink/NVSwitch topology (not auto-captured; set by adapter)",
    )

    # --- Layer 2: Software ---
    os_info: str | None = Field(default=None, description="OS and kernel version")
    python_version: str | None = Field(
        default=None, description="Python version string"
    )
    cuda_version: str | None = Field(default=None, description="CUDA version")
    framework_name: str | None = Field(
        default=None, description="Evaluation framework name"
    )
    framework_version: str | None = Field(
        default=None, description="Evaluation framework version"
    )
    container_image: str | None = Field(default=None, description="Container image URI")
    key_packages: dict[str, str] = Field(
        default_factory=dict, description="Package name to version mapping"
    )

    # --- Layer 3: Kubernetes ---
    k8s_pod_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Pod labels from K8s downward API (/etc/podinfo/labels)",
    )
    k8s_resource_limits: dict[str, str] = Field(
        default_factory=dict, description="K8s resource limits"
    )

    # --- Layer 4: Model identity ---
    model_id: str | None = Field(
        default=None, description="Model identifier (HF repo or endpoint URL)"
    )
    model_version: str | None = Field(
        default=None, description="Model version (SHA-256 or API version)"
    )
    model_provider: str | None = Field(
        default=None,
        description="Provider type: local | openai-compatible | huggingface",
    )

    # --- Layer 5: Run provenance ---
    collection_id: str | None = Field(
        default=None, description="Evaluation Collection identifier"
    )
    collection_version: str | None = Field(
        default=None, description="Semver version of the Collection"
    )
    scorer_ids: list[str] = Field(
        default_factory=list, description="Versioned scorer IDs applied"
    )
    dataset_hash: str | None = Field(
        default=None, description="SHA-256 hash of evaluation dataset(s)"
    )
    started_at: str | None = Field(
        default=None, description="ISO 8601 job start timestamp"
    )
    completed_at: str | None = Field(
        default=None, description="ISO 8601 job completion timestamp"
    )
    aggregate_results: dict[str, Any] = Field(
        default_factory=dict, description="Top-level metric results"
    )
    per_task_results: dict[str, Any] = Field(
        default_factory=dict, description="Per-task metric breakdown"
    )
    confidence_intervals: dict[str, Any] = Field(
        default_factory=dict, description="95% CIs per metric"
    )
    autograder_bias: dict[str, Any] = Field(
        default_factory=dict, description="Autograder bias estimates per scorer"
    )
    oci_artifact_ref: str | None = Field(
        default=None, description="OCI registry reference for the EvalCard artifact"
    )
    signature: str | None = Field(
        default=None, description="Sigstore Cosign signature of the EvalCard artifact"
    )
    generated_by: str | None = Field(
        default=None, description="EvalHub Server instance ID"
    )

    # --- Capture completeness ---
    capture_completeness: float | None = Field(
        default=None,
        description="Fraction of 26 spec fields that are non-null (0.0-1.0)",
    )

    # --- Escape hatch ---
    custom: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific fields"
    )

    @staticmethod
    def _capture_k8s_context() -> tuple[dict[str, str], dict[str, str], str | None]:
        """Best-effort Kubernetes context detection via downward API.

        Returns (pod_labels, resource_limits, container_image).
        All values default to empty/None if not running in a K8s pod.
        """
        import os
        from pathlib import Path

        pod_labels: dict[str, str] = {}
        resource_limits: dict[str, str] = {}
        container_image = os.environ.get("EVALHUB_CONTAINER_IMAGE")

        if not Path("/var/run/secrets/kubernetes.io/serviceaccount/token").exists():
            return pod_labels, resource_limits, container_image

        for env_key, limit_key in [
            ("MY_CPU_LIMIT", "cpu"),
            ("MY_MEM_LIMIT", "memory"),
            ("MY_GPU_LIMIT", "nvidia.com/gpu"),
        ]:
            val = os.environ.get(env_key)
            if val:
                resource_limits[limit_key] = val

        podinfo = Path("/etc/podinfo")
        if podinfo.exists():
            labels_file = podinfo / "labels"
            if labels_file.exists():
                try:
                    import shlex

                    for line in labels_file.read_text().strip().splitlines():
                        k, _, v = line.partition("=")
                        try:
                            pod_labels[k.strip()] = shlex.split(v.strip())[0]
                        except (ValueError, IndexError):
                            pod_labels[k.strip()] = v.strip().strip('"')
                except Exception:
                    logger.debug("Failed to read pod labels from %s", labels_file)

        return pod_labels, resource_limits, container_image

    def _compute_completeness(self) -> float:
        """Compute fraction of 26 spec fields that are non-null."""
        spec_fields = [
            self.gpu_model,
            self.gpu_count,
            self.gpu_driver_version,
            self.cpu_model,
            self.total_memory_gb,
            self.nvlink_topology,
            self.os_info,
            self.python_version,
            self.cuda_version,
            self.framework_name,
            self.framework_version,
            self.container_image,
            self.key_packages or None,
            self.k8s_pod_labels or None,
            self.k8s_resource_limits or None,
            self.model_id,
            self.model_version,
            self.model_provider,
            self.collection_id,
            self.collection_version,
            self.dataset_hash,
            self.started_at,
            self.completed_at,
            self.aggregate_results or None,
            self.confidence_intervals or None,
            self.oci_artifact_ref,
        ]
        populated = sum(1 for f in spec_fields if f is not None)
        return round(populated / _ENV_CARD_SPEC_FIELD_COUNT, 2)

    @classmethod
    def capture(
        cls,
        framework_name: str | None = None,
        framework_version: str | None = None,
        container_image: str | None = None,
        extra_packages: list[str] | None = None,
        **kwargs: Any,
    ) -> EnvironmentCardMetadata:
        """Auto-capture the current execution environment (Layers 1-3).

        Call at the start of ``run_benchmark_job()`` for zero-effort tracking.
        Layers 4-5 (model identity, run provenance) are populated by the
        EvalHub Server after job completion.

        Args:
            framework_name: Evaluation framework name (set explicitly).
            framework_version: Evaluation framework version string.
            container_image: Container image URI. Falls back to
                ``EVALHUB_CONTAINER_IMAGE`` env var.
            extra_packages: Additional package names to capture versions for.
            **kwargs: Added to ``custom`` dict.
        """
        import importlib.metadata
        import os

        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        cuda_version, gpu_model, gpu_count, gpu_driver = None, None, None, None
        cpu_model = None
        try:
            import torch

            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                gpu_names = {torch.cuda.get_device_name(i) for i in range(gpu_count)}
                gpu_model = ", ".join(sorted(gpu_names))
        except Exception:
            logger.debug("torch/CUDA detection skipped (not installed or unavailable)")

        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpu_driver = result.stdout.strip().splitlines()[0]
        except Exception:
            logger.debug("nvidia-smi detection skipped (not available)")

        try:
            cpu_model = platform.processor() or None
        except Exception:
            logger.debug("CPU model detection failed")

        base_packages = [
            "torch",
            "transformers",
            "vllm",
            "lm_eval",
            "garak",
            "ragas",
            "mlflow",
        ]
        all_packages = list(set(base_packages + (extra_packages or [])))
        key_packages: dict[str, str] = {}
        for pkg in all_packages:
            try:
                key_packages[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                pass

        try:
            os_info = platform.platform()
        except Exception:
            os_info = None
            logger.debug("OS info detection failed")

        k8s_pod_labels, k8s_resource_limits, k8s_image = cls._capture_k8s_context()
        container_image = (
            container_image or os.environ.get("EVALHUB_CONTAINER_IMAGE") or k8s_image
        )

        instance = cls(
            python_version=py_version,
            framework_name=framework_name,
            framework_version=framework_version,
            cuda_version=cuda_version,
            gpu_model=gpu_model,
            gpu_count=gpu_count,
            gpu_driver_version=gpu_driver,
            cpu_model=cpu_model,
            os_info=os_info,
            container_image=container_image,
            key_packages=key_packages,
            k8s_pod_labels=k8s_pod_labels,
            k8s_resource_limits=k8s_resource_limits,
            custom=kwargs,
        )
        instance.capture_completeness = instance._compute_completeness()
        return instance

"""Reproducible Harbor adapters for OpenHands benchmark runs."""

import asyncio
import json
import re
from pathlib import Path

from harbor.agents.installed.openhands_sdk import (  # pyright: ignore[reportMissingImports]
    OpenHandsSDK,
)
from harbor.environments.base import (  # pyright: ignore[reportMissingImports]
    BaseEnvironment,
)
from harbor.models.agent.context import (  # pyright: ignore[reportMissingImports]
    AgentContext,
)

from benchmark_agents.delegation import (
    terminal_benchmark_delegation_instructions,
    terminal_benchmark_single_agent_instructions,
)
from benchmark_agents.provenance import (
    OPENHANDS_SDK_SOURCE,
    openhands_sdk_source_commit,
)
from benchmarks.tracing.agentsight import AgentSightProfiler
from benchmarks.tracing.harbor import (
    HarborTraceAttempt,
    allocate_harbor_trace_attempt,
    attach_harbor_agentsight_profile,
    promote_harbor_trace_attempt,
    start_harbor_agentsight_profile,
    trace_agent_timeout_from_trial_config,
    trace_container_image_from_trial_config,
    trace_instance_id_from_trial_config,
)


OPENHANDS_SDK_INSTALL_ROOT = "/installed-agent/software-agent-sdk"
BENCHMARK_RUNTIME_ROOT = "/installed-agent/benchmark-runtime"
OPENHANDS_PYTHON = "/opt/openhands-sdk-venv/bin/python"
TRACE_CONFIG_ENV = "OPENHANDS_BENCHMARK_TRACE_CONFIG"


class ReproducibleOpenHandsSDK(OpenHandsSDK):
    """Install the exact vendored SDK and enforce one provider attempt."""

    enable_delegation = False

    def __init__(
        self,
        logs_dir: Path,
        prompt_template_path: Path | str | None = None,
        sdk_commit: str | None = None,
        trace_root: str | None = None,
        trace_run_id: str | None = None,
        trace_created_at: str | None = None,
        trace_benchmark: str | None = None,
        benchmark_commit: str | None = None,
        evaluation_workers: int = 1,
        benchmark_retries: int = 0,
        harbor_version: str = "unknown",
        *args,
        **kwargs,
    ) -> None:
        source_commit = openhands_sdk_source_commit()
        if sdk_commit != source_commit:
            raise ValueError(
                "sdk_commit must match the clean vendored OpenHands SDK revision "
                f"{source_commit}"
            )
        self._sdk_commit = source_commit
        trace_values = (
            trace_root,
            trace_run_id,
            trace_created_at,
            trace_benchmark,
            benchmark_commit,
        )
        if any(value is not None for value in trace_values) and not all(
            value is not None for value in trace_values
        ):
            raise ValueError("OpenHands Harbor tracing requires complete metadata")
        if benchmark_commit is not None and not re.fullmatch(
            r"[0-9a-f]{40}", benchmark_commit
        ):
            raise ValueError("benchmark_commit must be a full Git SHA")
        if trace_benchmark is not None and not trace_benchmark.strip():
            raise ValueError("trace_benchmark cannot be empty")
        if evaluation_workers < 1 or benchmark_retries < 0:
            raise ValueError("OpenHands Harbor trace execution metadata is invalid")
        self._trace_root = Path(trace_root).resolve() if trace_root else None
        self._trace_run_id = trace_run_id
        self._trace_created_at = trace_created_at
        self._trace_benchmark = trace_benchmark
        self._benchmark_commit = benchmark_commit
        self._evaluation_workers = evaluation_workers
        self._benchmark_retries = benchmark_retries
        self._harbor_version = harbor_version
        self._trace_attempt: HarborTraceAttempt | None = None
        self._agentsight_profiler: AgentSightProfiler | None = None
        if prompt_template_path is None:
            logs_dir.mkdir(parents=True, exist_ok=True)
            prompt_template_path = logs_dir / "single-agent-prompt.j2"
            prompt_template_path.write_text(
                "{{ instruction }}\n"
                f"{terminal_benchmark_single_agent_instructions()}\n",
                encoding="utf-8",
            )
        super().__init__(
            logs_dir=logs_dir,
            prompt_template_path=prompt_template_path,
            *args,
            **kwargs,
        )

    async def install(self, environment: BaseEnvironment) -> None:
        if self._trace_root is not None:
            self._trace_attempt = allocate_harbor_trace_attempt(
                trace_root=self._trace_root,
                instance_id=trace_instance_id_from_trial_config(self.logs_dir),
                agent_timeout_seconds=trace_agent_timeout_from_trial_config(
                    self.logs_dir
                ),
                container_image=trace_container_image_from_trial_config(self.logs_dir),
            )
        await super().install(environment)

        upload_roots = await environment.exec(
            command=(
                f"mkdir -p {OPENHANDS_SDK_INSTALL_ROOT} "
                f"{BENCHMARK_RUNTIME_ROOT}/benchmarks"
            ),
            user="root",
        )
        if upload_roots.return_code != 0:
            raise RuntimeError(
                "Failed to prepare OpenHands benchmark upload roots: "
                f"{upload_roots.stderr}"
            )

        for package in ("openhands-sdk", "openhands-tools"):
            await environment.upload_dir(
                source_dir=OPENHANDS_SDK_SOURCE / package,
                target_dir=f"{OPENHANDS_SDK_INSTALL_ROOT}/{package}",
            )

        if self._trace_attempt is not None:
            tracing_source = Path(__file__).parents[1] / "benchmarks" / "tracing"
            await environment.upload_dir(
                source_dir=tracing_source,
                target_dir=f"{BENCHMARK_RUNTIME_ROOT}/benchmarks/tracing",
            )
            existing_pythonpath = self._extra_env.get("PYTHONPATH", "")
            self._extra_env["PYTHONPATH"] = ":".join(
                value
                for value in (BENCHMARK_RUNTIME_ROOT, existing_pythonpath)
                if value
            )
            if not self.model_name or self.session_id is None:
                raise ValueError("Harbor did not initialize trace agent identity")
            self._extra_env[TRACE_CONFIG_ENV] = json.dumps(
                {
                    "run_id": self._trace_run_id,
                    "created_at": self._trace_created_at,
                    "benchmark": self._trace_benchmark,
                    "instance_id": self._trace_attempt.instance_id,
                    "attempt": self._trace_attempt.attempt,
                    "container_root": str(self._trace_attempt.container_root),
                    "benchmark_revision": self._benchmark_commit,
                    "framework_revision": self._sdk_commit,
                    "model": self.model_name,
                    "evaluation_workers": self._evaluation_workers,
                    "inference_timeout_seconds": (
                        self._trace_attempt.agent_timeout_seconds
                    ),
                    "benchmark_retries": self._benchmark_retries,
                    "session_id": self.session_id,
                    "harbor_version": self._harbor_version,
                    "container_image": self._trace_attempt.container_image,
                },
                separators=(",", ":"),
            )
        install_result = await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                'if [ -f "$HOME/.local/bin/env" ]; then '
                'source "$HOME/.local/bin/env"; fi; '
                "source /opt/openhands-sdk-venv/bin/activate; "
                "uv pip install --reinstall --no-deps "
                f"{OPENHANDS_SDK_INSTALL_ROOT}/openhands-sdk "
                f"{OPENHANDS_SDK_INSTALL_ROOT}/openhands-tools; "
                "python -c 'import openhands.sdk; "
                f'assert openhands.sdk.__version__ == "{self._version}"\''
            ),
        )
        if install_result.return_code != 0:
            raise RuntimeError(
                "Failed to install the exact vendored OpenHands SDK revision: "
                f"{install_result.stderr}"
            )

        move_result = await environment.exec(
            command=(
                "mv -f /installed-agent/run_agent.py /installed-agent/run_agent_base.py"
            ),
            user="root",
        )
        if move_result.return_code != 0:
            raise RuntimeError(
                f"Failed to preserve Harbor's OpenHands runner: {move_result.stderr}"
            )

        runner_source = Path(__file__).with_name("openhands_harbor_runner.py")
        runner_copy = self.logs_dir / "run_benchmark_agent.py"
        runner_copy.write_text(
            runner_source.read_text().replace(
                "ENABLE_DELEGATION = True",
                f"ENABLE_DELEGATION = {self.enable_delegation}",
                1,
            )
        )
        await environment.upload_file(
            source_path=runner_copy,
            target_path="/installed-agent/run_agent.py",
        )
        chmod_result = await environment.exec(
            command="chmod +x /installed-agent/run_agent.py",
            user="root",
        )
        if chmod_result.return_code != 0:
            raise RuntimeError(
                "Failed to make the benchmark OpenHands runner executable: "
                f"{chmod_result.stderr}"
            )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self._trace_attempt is None:
            await super().run(instruction, environment, context)
            return
        if self._trace_run_id is None or self._trace_benchmark is None:
            raise ValueError("Harbor did not initialize OpenHands trace metadata")
        self._agentsight_profiler = await asyncio.to_thread(
            start_harbor_agentsight_profile,
            logs_dir=self.logs_dir,
            trace_run_id=self._trace_run_id,
            benchmark=self._trace_benchmark,
            framework="openhands",
            attempt=self._trace_attempt,
            docker_session_id=environment.session_id,
            tls_python_path=OPENHANDS_PYTHON,
        )
        try:
            await super().run(instruction, environment, context)
        finally:
            await asyncio.to_thread(self._agentsight_profiler.finish)

    def populate_context_post_run(self, context: AgentContext) -> None:
        super().populate_context_post_run(context)
        if self._trace_attempt is None or self._trace_root is None:
            return
        metadata = {**(context.metadata or {})}
        try:
            if self._agentsight_profiler is None:
                raise ValueError("OpenHands AgentSight profiler was not initialized")
            attach_harbor_agentsight_profile(
                logs_dir=self.logs_dir,
                attempt=self._trace_attempt,
                profiler=self._agentsight_profiler,
            )
        except Exception as error:
            metadata["agentsight_profile"] = {
                "health": "failed",
                "error": type(error).__name__,
            }
            if (
                self._agentsight_profiler is not None
                and self._agentsight_profiler.strict
            ):
                raise
        try:
            destination = promote_harbor_trace_attempt(
                logs_dir=self.logs_dir,
                trace_root=self._trace_root,
                attempt=self._trace_attempt,
            )
            metadata["benchmark_trace"] = {
                "path": str(destination),
                "instance_id": self._trace_attempt.instance_id,
                "attempt": self._trace_attempt.attempt,
            }
            profile_path = destination / "profiles" / "agentsight"
            if profile_path.is_dir() and self._agentsight_profiler is not None:
                metadata["agentsight_profile"] = {
                    "path": str(profile_path),
                    "profile_id": self._agentsight_profiler.target.profile_id,
                }
        except Exception as error:
            metadata["benchmark_trace"] = {
                "health": "failed",
                "error": type(error).__name__,
            }
        context.metadata = metadata


class DelegatingOpenHandsSDK(ReproducibleOpenHandsSDK):
    """Enable native blocking subagent delegation on the reproducible runtime."""

    enable_delegation = True

    def __init__(
        self,
        logs_dir: Path,
        prompt_template_path: Path | str | None = None,
        *args,
        **kwargs,
    ) -> None:
        if prompt_template_path is None:
            logs_dir.mkdir(parents=True, exist_ok=True)
            prompt_template_path = logs_dir / "multiagent-prompt.j2"
            prompt_template_path.write_text(
                "{{ instruction }}\n"
                f"{terminal_benchmark_delegation_instructions()}\n"
            )
        super().__init__(
            logs_dir=logs_dir,
            prompt_template_path=prompt_template_path,
            *args,
            **kwargs,
        )

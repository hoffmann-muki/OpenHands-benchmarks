"""Reproducible Harbor adapters for OpenHands benchmark runs."""

from pathlib import Path

from harbor.agents.installed.openhands_sdk import (  # pyright: ignore[reportMissingImports]
    OpenHandsSDK,
)
from harbor.environments.base import (  # pyright: ignore[reportMissingImports]
    BaseEnvironment,
)

from benchmark_agents.delegation import terminal_benchmark_delegation_instructions
from benchmark_agents.provenance import (
    OPENHANDS_SDK_SOURCE,
    openhands_sdk_source_commit,
)


OPENHANDS_SDK_INSTALL_ROOT = "/installed-agent/software-agent-sdk"


class ReproducibleOpenHandsSDK(OpenHandsSDK):
    """Install the exact vendored SDK and enforce one provider attempt."""

    enable_delegation = False

    def __init__(
        self,
        logs_dir: Path,
        prompt_template_path: Path | str | None = None,
        sdk_commit: str | None = None,
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
        super().__init__(
            logs_dir=logs_dir,
            prompt_template_path=prompt_template_path,
            *args,
            **kwargs,
        )

    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)

        for package in ("openhands-sdk", "openhands-tools"):
            await environment.upload_dir(
                source_dir=OPENHANDS_SDK_SOURCE / package,
                target_dir=f"{OPENHANDS_SDK_INSTALL_ROOT}/{package}",
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

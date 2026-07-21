"""Harbor adapter that enables native OpenHands supervisor delegation."""

from pathlib import Path

from harbor.agents.installed.openhands_sdk import (  # pyright: ignore[reportMissingImports]
    OpenHandsSDK,
)
from harbor.environments.base import (  # pyright: ignore[reportMissingImports]
    BaseEnvironment,
)

from benchmark_agents.delegation import terminal_benchmark_delegation_instructions


class DelegatingOpenHandsSDK(OpenHandsSDK):
    """Install Harbor's SDK runner with blocking subagent delegation enabled."""

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

    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)

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
        runner_copy = self.logs_dir / "run_delegating_agent.py"
        runner_copy.write_text(runner_source.read_text())
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
                "Failed to make the delegating OpenHands runner executable: "
                f"{chmod_result.stderr}"
            )

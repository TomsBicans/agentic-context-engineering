from experiment_runner.models.config import RunConfig
from experiment_runner.models.enums import SystemName
from experiment_runner.runners.base import BaseRunner


# Populated lazily on first access to avoid importing heavy dependencies at
# module load time (e.g. importing the agent package triggers LangChain init).
_REGISTRY: dict[SystemName, type[BaseRunner]] | None = None


def _build_registry() -> dict[SystemName, type[BaseRunner]]:
    from experiment_runner.runners.ace import AceRunner
    from experiment_runner.runners.baseline.anythingllm import AnythingLLMRunner
    from experiment_runner.runners.baseline.claudecodelocal import ClaudeCodeLocalRunner
    from experiment_runner.runners.baseline.clawcode import ClawCodeRunner
    from experiment_runner.runners.baseline.openclaw import OpenClawRunner
    from experiment_runner.runners.manual import ManualRunner

    return {
        SystemName.ACE: AceRunner,
        SystemName.CLAUDE_CODE_CLOUD: ManualRunner,
        SystemName.CLAUDE_CODE_LOCAL: ClaudeCodeLocalRunner,
        SystemName.CHATGPT_CODEX: ManualRunner,
        SystemName.CLAWCODE: ClawCodeRunner,
        SystemName.ANYTHINGLLM: AnythingLLMRunner,
        SystemName.OPENCLAW: OpenClawRunner,
        SystemName.OPEN_WEBUI: ManualRunner,
        SystemName.PRIVATEGPT: ManualRunner,
        SystemName.PERPLEXITY: ManualRunner,
    }


def get_runner(config: RunConfig) -> BaseRunner:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()

    runner_cls = _REGISTRY.get(config.system)
    if runner_cls is None:
        raise ValueError(f"No runner registered for system: {config.system!r}")
    return runner_cls(config)

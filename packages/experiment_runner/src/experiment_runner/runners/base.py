from abc import ABC, abstractmethod

from experiment_runner.models.config import RunConfig
from experiment_runner.models.question import Question
from experiment_runner.models.result import RunResult


class BaseRunner(ABC):
    """Contract that every system-specific runner must satisfy.

    Runners are instantiated once per experiment session (so that expensive
    initialisation such as loading a local model happens only once) and then
    called repeatedly via `run()`.
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config

    def setup(self) -> None:
        """Called once before the question batch begins.

        Override to perform expensive one-time initialisation (e.g. starting
        a Docker container, authenticating to an API). Default is a no-op.
        """

    def teardown(self) -> None:
        """Called once after the question batch ends (even on error).

        Override to release resources acquired in setup(). Default is a no-op.
        """

    @abstractmethod
    def run(self, question: Question) -> RunResult:
        """Execute a single question against the system under test.

        Must always return a RunResult. On failure, populate `answer_error`
        rather than raising — the caller is responsible for deciding whether
        to abort the batch.
        """
        ...

    def _base_result(self, question: Question) -> RunResult:
        """Construct a RunResult pre-populated with identity fields from config."""
        return RunResult(
            system_name=self.config.system,
            automation_level=self.config.automation_level,
            corpus=self.config.corpus,
            question_id=question.id,
            question_text=question.question,
            model=self.config.model,
            quantization=self.config.quantization,
            inference_config=self.config.inference_config,
            reasoning_enabled=self.config.reasoning_enabled,
        )

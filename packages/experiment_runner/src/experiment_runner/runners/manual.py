from experiment_runner.models.enums import AutomationLevel
from experiment_runner.models.question import Question
from experiment_runner.models.result import RunResult
from experiment_runner.runners.base import BaseRunner


class ManualRunner(BaseRunner):
    """Runner for systems that require manual interaction (e.g. web UI, no API).

    Prints the question to stdout, reads the answer from stdin, and records it.
    All metric fields that require programmatic access are left as None.
    """

    def run(self, question: Question) -> RunResult:
        result = self._base_result(question)
        result.automation_level = AutomationLevel.MANUAL

        print(f"\n[MANUAL] System: {self.config.system.value}")
        print(f"Corpus:  {self.config.corpus.value}")
        print(f"Q [{question.id}]: {question.question}")
        print("Paste the system's answer, then press Enter twice:")

        lines: list[str] = []
        while True:
            line = input()
            if not line and lines:
                break
            lines.append(line)

        result.answer_text = "\n".join(lines).strip() or None
        return result

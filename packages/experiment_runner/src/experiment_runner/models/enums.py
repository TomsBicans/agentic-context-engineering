from enum import Enum


class SystemName(str, Enum):
    ACE = "ace"
    CLAUDE_CODE_CLOUD = "claude_code_cloud"
    CLAUDE_CODE_LOCAL = "claude_code_local"
    CHATGPT_CODEX = "chatgpt_codex"
    CLAWCODE = "clawcode"
    ANYTHINGLLM = "anythingllm"
    OPENCLAW = "openclaw"
    OPEN_WEBUI = "open_webui"
    PRIVATEGPT = "privategpt"
    PERPLEXITY = "perplexity"


class Corpus(str, Enum):
    OBLIVION_WIKI = "oblivion_wiki"
    SOLAR_SYSTEM_WIKI = "solar_system_wiki"
    SCIPY = "scipy"


class AutomationLevel(str, Enum):
    # Fully programmatic — no human interaction required during the run.
    FULL = "full"
    # Some manual steps required (e.g. copy-paste into a GUI, confirm a prompt).
    PARTIAL = "partial"
    # Results entered by hand after manual interaction with the system.
    MANUAL = "manual"


class CitationQuality(str, Enum):
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    # System does not produce citations at all.
    NOT_APPLICABLE = "not_applicable"

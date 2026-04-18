import re
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from cli.ui.rich_render import console, render_error, render_statements, render_stream_live

SLASH_COMMANDS = ["/help", "/set", "/exit", "/quit"]

WELCOME_BANNER = """[bold cyan]
  ╔═══════════════════════════════════════╗
  ║   ace  — corpus information retrieval ║
  ╚═══════════════════════════════════════╝
[/bold cyan]
Type a question to query the corpus.
Slash commands: [bold]/help[/bold]  [bold]/set[/bold]  [bold]/exit[/bold]
"""

HELP_TEXT = """[bold]Slash commands[/bold]
  [cyan]/help[/cyan]              Show this help message
  [cyan]/set k <n>[/cyan]         Set max statements to display (default 10)
  [cyan]/set model <name>[/cyan]  Change the LLM model
  [cyan]/exit[/cyan]  [cyan]/quit[/cyan]    Quit the REPL

[bold]Query tips[/bold]
  Just type your question and press Enter.
  Use Ctrl-C or Ctrl-D to exit.
"""


def run_repl(args) -> None:
    history_file = Path.home() / ".ace_history"
    session: PromptSession = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=WordCompleter(SLASH_COMMANDS, pattern=re.compile(r"/")),
        multiline=False,
    )
    console.print(WELCOME_BANNER)

    agent = None
    k: int = getattr(args, "k", 10)
    model: str = getattr(args, "model", "qwen3:4b")
    num_ctx: int = getattr(args, "num_ctx", 8192)
    time_limit: int = getattr(args, "time_limit", 60)
    require_tools: bool = getattr(args, "require_tools", True)
    reasoning_enabled: bool = getattr(args, "reasoning_enabled", False)
    path_to_corpora_str = getattr(args, "path_to_corpora", None)
    corpora_path = Path(path_to_corpora_str) if path_to_corpora_str else Path.cwd()

    while True:
        try:
            with patch_stdout():
                text = session.prompt("> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not text:
            continue
        if text in ("/exit", "/quit"):
            console.print("[dim]Bye.[/dim]")
            break
        if text == "/help":
            console.print(HELP_TEXT)
            continue
        if text.startswith("/set "):
            prev_model = model
            k, model = _handle_set(text, k, model)
            if model != prev_model and agent is not None:
                agent = None
                console.print("[dim]Agent will reinitialize with the new model on next query.[/dim]")
            continue

        # Real query — lazy-init agent on first use
        if agent is None:
            if not corpora_path.exists():
                render_error(f"Corpora path does not exist: {corpora_path}")
                continue
            from agent.core import AgentRole, initialize_agent
            with console.status("[bold green]Initializing agent…"):
                try:
                    agent = initialize_agent(
                        llm_model=model,
                        role=AgentRole.EXAMINEE,
                        path_to_corpora=corpora_path,
                        temperature=0.03,
                        num_ctx=num_ctx,
                        time_limit=time_limit,
                        enforce_tools=require_tools,
                        reasoning_enabled=reasoning_enabled,
                    )
                except Exception as exc:
                    render_error(f"Failed to initialize agent: {exc}")
                    continue

        try:
            response = render_stream_live(agent, text)
            if response is not None:
                render_statements(response, k=k, fmt="table")
        except Exception as exc:
            render_error(f"Query failed: {exc}")


def _handle_set(text: str, k: int, model: str) -> tuple[int, str]:
    """Handle /set <key> <value> commands. Returns updated (k, model)."""
    parts = text.split(maxsplit=2)
    if len(parts) < 3:
        console.print("[yellow]Usage: /set <key> <value>[/yellow]")
        return k, model

    key, value = parts[1], parts[2]
    if key == "k":
        try:
            k = int(value)
            console.print(f"[green]k set to {k}[/green]")
        except ValueError:
            console.print(f"[yellow]Invalid value for k: {value!r}[/yellow]")
    elif key == "model":
        model = value
        console.print(f"[green]model set to {model!r}[/green]")
    else:
        console.print(f"[yellow]Unknown setting: {key!r}. Valid keys: k, model[/yellow]")
    return k, model

from pathlib import Path

from agent.core import AgentRole, initialize_agent
from agent.interface.invoke import invoke_agent
from cli.ui.rich_render import console, render_error, render_statements, render_stream_live


def run_query(args) -> None:
    corpora_path = Path(args.path_to_corpora) if args.path_to_corpora else Path.cwd()

    if not corpora_path.exists():
        render_error(f"Path to corpora does not exist: {corpora_path}")
        raise SystemExit(1)

    with console.status("[bold green]Initializing agent…"):
        try:
            agent = initialize_agent(
                llm_model=args.model,
                role=AgentRole.EXAMINEE,
                path_to_corpora=corpora_path,
                temperature=0.03,
                num_ctx=args.num_ctx,
                time_limit=args.time_limit,
                enforce_tools=args.require_tools,
                reasoning_enabled=args.reasoning_enabled,
            )
        except Exception as exc:
            render_error(f"Failed to initialize agent: {exc}")
            raise SystemExit(1)

    if args.stream:
        response = render_stream_live(agent, args.prompt)
    else:
        response = invoke_agent(agent, args.prompt)

    if response is None:
        render_error("Agent returned no response.")
        raise SystemExit(1)

    render_statements(response, k=args.k, fmt=args.format)

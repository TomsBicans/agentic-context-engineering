import argparse


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="ace",
        description="Corpus information retrieval CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    q = subparsers.add_parser("query", help="Single query against a corpus")
    q.add_argument("prompt", help="Question to ask the agent")
    q.add_argument("--model", default="qwen3:4b")
    q.add_argument("--num-ctx", type=int, default=8192, dest="num_ctx")
    q.add_argument("--path-to-corpora", default=None, dest="path_to_corpora")
    q.add_argument("--k", type=int, default=10)
    q.add_argument("--format", choices=["table", "md", "json"], default="table", dest="format")
    q.add_argument("--time-limit", type=int, default=60, dest="time_limit")
    q.add_argument("--stream", dest="stream", action="store_true", default=True)
    q.add_argument("--no-stream", dest="stream", action="store_false")
    q.add_argument("--require-tools", dest="require_tools", action="store_true", default=True)
    q.add_argument("--no-require-tools", dest="require_tools", action="store_false")
    q.add_argument("--reasoning-enabled", dest="reasoning_enabled", action="store_true", default=False)

    args = parser.parse_args(argv)

    if args.command == "query":
        from cli.commands.query import run_query
        run_query(args)
    else:
        from cli.repl.session import run_repl
        run_repl(args)


if __name__ == "__main__":
    main()


run_tests:
	py -m pytest -q


agent_h:
	py -m src.agent.core -h

agent:
	py -m src.agent.core

main_h:
	py -m src.cli.main -h

main:
	py -m src.cli.main

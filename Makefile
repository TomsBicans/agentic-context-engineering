
run_tests:
	py -m pytest -q


agent_h:
	py -m src.agent.core -h

model=gemma3:4b # no tool call support
model=deepseek-r1:8b # no tool call support

model=qwen3:8b
model=llama3.1:8b
model=cogito:8b
model=functiongemma:270m
model=mistral-nemo:12b
model=qwen3:14b
model=qwen3:4b

q="how many Extreme trans-Neptunian objects are there?"
q="What is the Sun's mass?"
q="Where can i find the list of minor planets?"
q="What is the Earth's mass?"
q="How many commets are in the solar system?"

ctx=2048
ctx=4096
ctx=8192
ctx=32768
ctx=16384

agent_test:
	py -m src.agent.core ${q} --model ${model} --num_ctx 8192 --role examinee --path-to-corpora "./corpora/scraped_data/solar_system_wiki"

agent:
	py -m src.agent.core

main_h:
	py -m src.cli.main -h

main:
	py -m src.cli.main

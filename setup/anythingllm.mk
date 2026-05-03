.PHONY: anythingllm-cli-install

anythingllm-cli-install:
	@echo "=== Installing AnythingLLM CLI ==="
	@if command -v any >/dev/null 2>&1; then \
		echo "AnythingLLM CLI is already installed:"; \
		any --version; \
	else \
		curl -fsSL https://raw.githubusercontent.com/Mintplex-Labs/anything-llm-cli/main/install.sh | sh; \
		echo "✅ AnythingLLM CLI installed successfully"; \
	fi


ANYTHINGLLM_CONTAINER := anythingllm-thesis
ANYTHINGLLM_STORAGE   := $(HOME)/anythingllm-thesis-data

anythingllm-up:
	mkdir -p $(ANYTHINGLLM_STORAGE)
	touch $(ANYTHINGLLM_STORAGE)/.env
	docker run -d \
		--name $(ANYTHINGLLM_CONTAINER) \
		-p 3001:3001 \
		--add-host=host.docker.internal:host-gateway \
		--cap-add SYS_ADMIN \
		-v $(ANYTHINGLLM_STORAGE):/app/server/storage \
		-v $(ANYTHINGLLM_STORAGE)/.env:/app/server/.env \
		-e STORAGE_DIR="/app/server/storage" \
		-e LLM_PROVIDER=ollama \
		-e OLLAMA_MODEL_PREF=qwen3:8b \
		-e OLLAMA_BASE_PATH=http://host.docker.internal:11434 \
		--restart unless-stopped \
		mintplexlabs/anythingllm:latest
	@echo "AnythingLLM started → http://localhost:3001"

anythingllm-down:
	docker stop $(ANYTHINGLLM_CONTAINER) 2>/dev/null || true
	docker rm $(ANYTHINGLLM_CONTAINER) 2>/dev/null || true


anythingllm-status:
	docker ps --filter name=$(ANYTHINGLLM_CONTAINER)


anythingllm-cli-test:
	any prompt "Reply with only the word TEST" -w default
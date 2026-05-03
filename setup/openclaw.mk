OPENCLAW_BIN := $(HOME)/.local/bin/openclaw

.PHONY: openclaw-install openclaw-up openclaw-down openclaw-status openclaw-clean openclaw-test

openclaw-install:
	@echo "=== Installing OpenClaw ==="
	@if command -v openclaw >/dev/null 2>&1; then \
		echo "OpenClaw is already installed:"; \
		openclaw --version; \
	else \
		echo "Installing OpenClaw via official script..."; \
		curl -fsSL https://openclaw.ai/install.sh | bash; \
		echo "OpenClaw installed successfully"; \
	fi

openclaw-up:
	@echo "=== Starting OpenClaw gateway ==="
	# 'restart' works even when the service is stopped; 'start' is a no-op if stopped on WSL2 systemd.
	openclaw gateway restart || openclaw gateway run &
	@echo "OpenClaw gateway started. Check status with: make openclaw-status"

openclaw-down:
	@echo "=== Stopping OpenClaw ==="
	openclaw gateway stop 2>/dev/null || true
	@echo "OpenClaw gateway stopped"

openclaw-status:
	@echo "=== OpenClaw Status ==="
	openclaw status || echo "OpenClaw not responding"

openclaw-clean:
	make openclaw-down
	rm -rf ~/.openclaw
	@echo "OpenClaw data cleaned"

openclaw-test:
	@echo "=== Testing OpenClaw CLI ==="
	openclaw agent --agent main --message "Reply with only the word TEST" --json
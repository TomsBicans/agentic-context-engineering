CLAWCODE_REPO := https://github.com/ultraworkers/claw-code.git
CLAWCODE_VERSION := v0.1.0
CLAWCODE_DIR := $(HOME)/.local/share/claw-code
CLAWCODE_BIN := $(HOME)/.local/bin/claw

.PHONY: clawcode-install clawcode-doctor clawcode-status clawcode-clean

clawcode-install:
	@echo "=== Installing ClawCode $(CLAWCODE_VERSION) ==="
	@if [ -x "$(CLAWCODE_BIN)" ]; then \
		echo "ClawCode is already installed:"; \
		$(CLAWCODE_BIN) version; \
	else \
		echo "Cloning and building ClawCode... (this may take a minute)"; \
		rm -rf $(CLAWCODE_DIR) && \
		mkdir -p $$(dirname $(CLAWCODE_DIR)) && \
		git clone --depth 1 --branch $(CLAWCODE_VERSION) $(CLAWCODE_REPO) $(CLAWCODE_DIR) && \
		cd $(CLAWCODE_DIR) && \
		./install.sh && \
		mkdir -p $(HOME)/.local/bin && \
		ln -sf $$(pwd)/rust/target/debug/claw $(CLAWCODE_BIN) && \
		echo "ClawCode installed successfully at $(CLAWCODE_BIN)"; \
	fi


clawcode-doctor:
	@$(CLAWCODE_BIN) doctor 2>/dev/null || echo "claw doctor failed or binary not found"

clawcode-status:
	@if [ -x "$(CLAWCODE_BIN)" ]; then \
		$(CLAWCODE_BIN) version; \
	else \
		echo "ClawCode not found in $(CLAWCODE_BIN)"; \
	fi

clawcode-clean:
	@rm -rf $(CLAWCODE_DIR)
	@rm -f $(CLAWCODE_BIN)

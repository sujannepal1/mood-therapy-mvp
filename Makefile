.DEFAULT_GOAL := help

BACKEND_DIR := backend

# ─────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show all available commands
	@echo ""
	@echo "  Music Mood Therapy — Available Commands"
	@echo "  ──────────────────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────

.PHONY: install
install: ## Install all backend Python dependencies via uv
	cd $(BACKEND_DIR) && uv sync

# ─────────────────────────────────────────────────────────────
# Nepali data pipeline (primary — what the app runs on)
# ─────────────────────────────────────────────────────────────

.PHONY: collect
collect: ## Download only NEW Nepali songs from Spotify playlists → appends to nepali_dataset.csv
	cd $(BACKEND_DIR) && uv run python src/collect_spotify_data.py

.PHONY: process
process: ## Convert teammate's features.csv → nepali_dataset_500.csv (303 rule-labeled songs)
	cd $(BACKEND_DIR) && uv run python src/process_features_csv.py

.PHONY: refresh
refresh: ## Full refresh: checks all 6 playlists → new songs only → deletes old models → retrains
	$(MAKE) collect
	@echo ""
	@echo "  Removing old models before retraining..."
	@rm -f $(BACKEND_DIR)/models/emotion_classifier_rf_nepali.pkl
	@rm -f $(BACKEND_DIR)/models/emotion_classifier_mlp_nepali.pkl
	@echo "  Done. Training fresh on updated dataset..."
	@echo ""
	$(MAKE) train

.PHONY: train
train: ## Train RF + MLP on combined Nepali datasets → *_nepali.pkl (run after collect + process)
	cd $(BACKEND_DIR) && uv run python src/train_classifier.py

# ─────────────────────────────────────────────────────────────
# Kaggle data pipeline (original — baseline comparison models only)
# ─────────────────────────────────────────────────────────────

.PHONY: label
label: ## Label 89k Kaggle songs with emotions → dataset_labeled.csv (requires dataset.csv)
	cd $(BACKEND_DIR) && uv run python src/label_emotions.py

.PHONY: train-kaggle
train-kaggle: ## Train RF + MLP on Kaggle labeled data (for comparison only, not used in production)
	$(MAKE) label
	cd $(BACKEND_DIR) && uv run python src/train_classifier.py

# ─────────────────────────────────────────────────────────────
# Run the app
# ─────────────────────────────────────────────────────────────

.PHONY: run
run: ## Run the CLI conversation app (Nepali models must exist — run 'make train' first)
	cd $(BACKEND_DIR) && uv run python src/pipeline.py

.PHONY: start
start: ## Smart start — trains Nepali models if missing, then runs the CLI
	@if [ ! -f $(BACKEND_DIR)/models/emotion_classifier_mlp_nepali.pkl ]; then \
		echo ""; \
		echo "  Nepali models not found — training now..."; \
		echo "  This takes ~1 minute and only happens once."; \
		echo ""; \
		$(MAKE) train; \
	fi
	$(MAKE) run

.PHONY: api
api: ## Start the FastAPI backend on port 8000 (hot-reload, Nepali models must exist)
	cd $(BACKEND_DIR) && uv run uvicorn src.api:app --reload --port 8000

.PHONY: frontend
frontend: ## Start the React + Vite dev server on port 5173
	cd frontend && npm run dev

.PHONY: typecheck
typecheck: ## Run TypeScript strict type-check on the frontend
	cd frontend && npm run typecheck

# ─────────────────────────────────────────────────────────────
# Docker (full stack)
# ─────────────────────────────────────────────────────────────

.PHONY: up
up: ## Build and start all services in Docker (backend :8000 + frontend :5173)
	docker compose up --build

.PHONY: down
down: ## Stop and remove all Docker containers
	docker compose down

# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────

.PHONY: demo
demo: ## Librosa audio demo — usage: make demo FILE=path/to/song.mp3
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make demo FILE=path/to/song.mp3"; \
		exit 1; \
	fi
	cd $(BACKEND_DIR) && uv run python demo/librosa_demo.py $(FILE)

# ─────────────────────────────────────────────────────────────
# Maintenance
# ─────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove all generated data and model files (forces full retrain)
	@echo "Removing generated files..."
	@rm -f $(BACKEND_DIR)/data/processed/dataset_labeled.csv
	@rm -f $(BACKEND_DIR)/data/processed/nepali_dataset_500.csv
	@rm -f $(BACKEND_DIR)/models/*.pkl
	@echo "Done. Run 'make train' to regenerate Nepali models."

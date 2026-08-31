# Run INSIDE the container (see root Makefile: `make up-rtfm`).
# From there: cd RTFM && make train

# Dataset e features (montadas em /data pelo docker-compose).
DATASET ?= shanghai
FEATURES_DIR ?= /data/shanghaitech/features/i3d-10crop
LIST_DIR ?= list/local
BATCH_SIZE ?= 32
MAX_EPOCH ?= 15000

.PHONY: make-lists train train-local

# Reaponta as lists do repo (caminhos absolutos dos autores) para FEATURES_DIR.
make-lists:
	python scripts/make_lists.py \
		--dataset "$(DATASET)" \
		--train-features-dir "$(FEATURES_DIR)/train" \
		--test-features-dir "$(FEATURES_DIR)/test" \
		--out-dir "$(LIST_DIR)"

train:
	python -m visdom.server & sleep 3 && python main.py

# Treino com as features locais. Rode `make make-lists` antes.
train-local:
	python -m visdom.server & sleep 3 && python main.py \
		--dataset "$(DATASET)" \
		--rgb-list "$(LIST_DIR)/$(if $(filter ucf,$(DATASET)),ucf-i3d.list,shanghai-i3d-train-10crop.list)" \
		--test-rgb-list "$(LIST_DIR)/$(if $(filter ucf,$(DATASET)),ucf-i3d-test.list,shanghai-i3d-test-10crop.list)" \
		--batch-size $(BATCH_SIZE) \
		--max-epoch $(MAX_EPOCH)

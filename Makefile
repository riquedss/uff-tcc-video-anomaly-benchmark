COMPOSE = docker compose

ifdef GPU
COMPOSE += -f docker-compose.yml -f docker-compose.gpu.yml
endif

.PHONY: all build build-anomaly build-pel4vad build-rtfm \
        up up-anomaly up-pel4vad up-rtfm \
        check-gpu \
        down clean

all: build

build: build-anomaly build-pel4vad build-rtfm

build-anomaly:
	$(COMPOSE) build anomaly-detection --no-cache

build-pel4vad:
	$(COMPOSE) build pel4vad --no-cache

build-rtfm:
	$(COMPOSE) build rtfm --no-cache

up-anomaly:
	$(COMPOSE) run --rm anomaly-detection bash

up-pel4vad:
	$(COMPOSE) run --rm pel4vad bash

up-rtfm:
	$(COMPOSE) run --rm --service-ports rtfm bash

# Confirma que cada container enxerga a GPU. Use com GPU=1:
#   make GPU=1 check-gpu
check-gpu:
	@for svc in anomaly-detection pel4vad rtfm; do \
		echo "--- $$svc ---"; \
		$(COMPOSE) run --rm $$svc python -c \
			'import torch; print("cuda:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")'; \
	done

up:
	$(COMPOSE) up

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down --rmi local --volumes --remove-orphans

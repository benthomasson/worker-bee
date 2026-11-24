IMAGE := ghcr.io/benthomasson/worker-bee-batch-fix
TAG := latest
PLATFORMS := linux/arm64

.PHONY: build rebuild push run clean

build:
	podman build --platform $(PLATFORMS) --manifest $(IMAGE):$(TAG) .

rebuild:
	podman manifest rm $(IMAGE):$(TAG) 2>/dev/null || true
	podman build --no-cache --platform $(PLATFORMS) --manifest $(IMAGE):$(TAG) .

push: build
	podman manifest push $(IMAGE):$(TAG) docker://$(IMAGE):$(TAG)

# Test locally: provide repos and API key
run:
	podman run -it \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e REPO_URL=https://github.com/benthomasson/worker-bee \
		-e EXPERT_URL=https://github.com/benthomasson/worker-bee-expert \
		-e TASK_TYPE=batch-fix \
		-e BELIEF_IDS="$(BELIEF_IDS)" \
		-e MODEL=openai:gpt-5.6-luna \
		$(IMAGE):$(TAG)

# Fix a single belief locally
fix:
	podman run -it \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e REPO_URL=https://github.com/benthomasson/worker-bee \
		-e EXPERT_URL=https://github.com/benthomasson/worker-bee-expert \
		-e TASK_TYPE=fix-belief \
		-e BELIEF_ID="$(BELIEF_ID)" \
		-e MODEL=openai:gpt-5.6-luna \
		$(IMAGE):$(TAG)

clean:
	podman rmi $(IMAGE):$(TAG) 2>/dev/null || true

IMAGE := ghcr.io/benthomasson/worker-bee-batch-fix
TAG := latest
PLATFORMS := linux/amd64

# Local paths — override for your setup
REPO_PATH ?= $(HOME)/git/worker-bee
EXPERT_PATH ?= $(HOME)/git/worker-bee-expert

.PHONY: build rebuild push run fix run-local fix-local verify-local clean

build:
	podman build --platform $(PLATFORMS) --manifest $(IMAGE):$(TAG) .

rebuild:
	podman manifest rm $(IMAGE):$(TAG) 2>/dev/null || true
	podman build --no-cache --platform $(PLATFORMS) --manifest $(IMAGE):$(TAG) .

push: build
	podman manifest push $(IMAGE):$(TAG) docker://$(IMAGE):$(TAG)

# ─── Clone mode: repos cloned at runtime (needs GH_TOKEN for private repos) ───

run:
	podman run -it \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e REPO_URL=https://github.com/benthomasson/worker-bee \
		-e EXPERT_URL=https://github.com/benthomasson/worker-bee-expert \
		-e TASK_TYPE=batch-fix \
		-e BELIEF_IDS="$(BELIEF_IDS)" \
		-e MODEL=openai:gpt-5.6-luna \
		$(IMAGE):$(TAG)

fix:
	podman run -it \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e REPO_URL=https://github.com/benthomasson/worker-bee \
		-e EXPERT_URL=https://github.com/benthomasson/worker-bee-expert \
		-e TASK_TYPE=fix-belief \
		-e BELIEF_ID="$(BELIEF_ID)" \
		-e MODEL=openai:gpt-5.6-luna \
		$(IMAGE):$(TAG)

# ─── Mount mode: bind-mount local repos (no credentials needed, faster) ───

run-local:
	podman run -it \
		-v $(REPO_PATH):/home/bee/workspace/repo \
		-v $(EXPERT_PATH):/home/bee/workspace/expert \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e TASK_TYPE=batch-fix \
		-e BELIEF_IDS="$(BELIEF_IDS)" \
		-e MODEL=openai:gpt-5.6-luna \
		$(IMAGE):$(TAG)

fix-local:
	podman run -it \
		-v $(REPO_PATH):/home/bee/workspace/repo \
		-v $(EXPERT_PATH):/home/bee/workspace/expert \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e TASK_TYPE=fix-belief \
		-e BELIEF_ID="$(BELIEF_ID)" \
		-e MODEL=openai:gpt-5.6-luna \
		$(IMAGE):$(TAG)

verify-local:
	podman run -it \
		-v $(REPO_PATH):/home/bee/workspace/repo \
		-v $(EXPERT_PATH):/home/bee/workspace/expert \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e TASK_TYPE=verify \
		-e BELIEF_ID="$(BELIEF_ID)" \
		-e MODEL=openai:gpt-5.6-luna \
		$(IMAGE):$(TAG)

clean:
	podman rmi $(IMAGE):$(TAG) 2>/dev/null || true

FROM quay.io/centos/centos:stream9

ARG BEE_USER=bee
ARG BEE_UID=1001
ARG BEE_GID=1006

# System packages
RUN dnf install -y --allowerasing \
        python3 python3-pip \
        git \
        curl which procps-ng \
    && dnf clean all

# Create bee user with matching host UID/GID for mount mode
RUN groupadd -g ${BEE_GID} ${BEE_USER} \
    && useradd -m -s /bin/bash -u ${BEE_UID} -g ${BEE_GID} ${BEE_USER}

# Install uv + worker-bee + reasonsforge as bee user
USER ${BEE_USER}
WORKDIR /home/${BEE_USER}

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN export PATH=$HOME/.local/bin:$PATH \
    && uv python install 3.14 \
    && uv tool install --python 3.14 "reasonsforge[openai,forge] @ git+https://github.com/benthomasson/reasonsforge" \
    && uv tool install --python 3.14 "worker-bee[openai] @ git+https://github.com/benthomasson/worker-bee"

RUN mkdir -p /home/${BEE_USER}/workspace

USER root

# Git identity
RUN git config --system user.email 'bee@worker-bee' \
    && git config --system user.name 'Worker Bee' \
    && git config --system init.defaultBranch main

RUN chmod 755 /home/${BEE_USER}

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER ${BEE_USER}
WORKDIR /home/${BEE_USER}/workspace

ENTRYPOINT ["/entrypoint.sh"]

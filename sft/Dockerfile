FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
git \
curl \
ca-certificates \
&& rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

RUN git clone https://github.com/deathbyknowledge/shellm /app/shellm
WORKDIR /app/shellm

RUN uv sync
RUN uv sync --extra sft --no-build-isolation

CMD ["/bin/bash"]
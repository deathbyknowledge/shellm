# Use the specified base image
FROM phusion/baseimage:noble-1.0.2

# Set non-interactive frontend to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install common tools and clean up in a single layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps coreutils findutils grep sed gawk tar git curl jq \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set default command to keep container running (phusion/baseimage uses my_init)
CMD ["/sbin/my_init"]
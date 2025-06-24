# Use the specified base image
FROM phusion/baseimage:noble-1.0.2

# Set non-interactive frontend to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# --- Create the 'shellm' user with root privileges (UID 0) ---
# By setting the UID to 0, 'shellm' becomes an alias for the root user.
# This grants full privileges without needing to use the 'sudo' command.
# -o allows creating a user with a non-unique UID.
# -u 0 sets the user ID to 0 (root).
# -g 0 sets the group ID to 0 (root).
# -m creates the user's home directory.
# -s sets the default shell.
RUN useradd --no-log-init -o -u 0 -g 0 -m -s /bin/bash shellm

# --- Install common tools ---
# Install common tools and clean up in a single layer. Sudo is not needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps coreutils findutils grep sed gawk tar git curl jq \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# --- Switch to the 'shellm' user context ---
# Set the home directory environment variable for future commands
ENV HOME=/home/shellm

# Set the working directory to the new user's home directory.
# Any subsequent RUN, CMD, or ENTRYPOINT instructions will be executed here.
WORKDIR /home/shellm

# Switch the context to our new user.
# Even though 'shellm' has UID 0, this sets the username for the shell.
USER shellm

# Set default command to keep container running.
# The `my_init` process will run as root (UID 0), which is now 'shellm'.
CMD ["/sbin/my_init"]

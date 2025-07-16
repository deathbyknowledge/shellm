import docker
import time
import socket

class Sandbox:
    """Manages an isolated Docker container with a persistent shell session."""
    
    def __init__(self, image="shellm-sandbox:latest", setup_commands=[]):
        self.image = image
        self.client = docker.from_env()
        self.container = None
        self.socket = None
        self.command_id = 0
        self.setup_commands = " && ".join(setup_commands).replace("'", "'\\''")
        print(setup_commands)

    def start(self):
        """Starts a new Docker container and sets up a persistent shell session."""
        print("Starting secure sandbox...")
        try:
            # Start container with bash as the main process
            self.container = self.client.containers.run(
                self.image,
                command="/bin/bash",
                tty=True,
                stdin_open=True,
                detach=True
            )
            # Install tools using exec_run
            print("Installing tools in sandbox...")
            exit_code, (stdout, stderr) = self.container.exec_run(
                f"/bin/bash -c '{self.setup_commands}'", demux=True
            )
            if exit_code != 0:
                raise Exception(f"Sandbox setup failed: {stderr.decode() if stderr else 'Unknown error'}")
            print("Sandbox ready.")
            # Attach to the bash process
            self.socket = self.container.attach_socket(
                params={'stdin': 1, 'stdout': 1, 'stderr': 1, 'stream': 1}
            )
            self.socket._sock.settimeout(1)  # Set timeout for socket reads
            self.socket._sock.send(b'stty -echo\n') # Don't echo input
            time.sleep(0.1)

        except Exception as e:
            print(f"Error starting sandbox: {e}")
            self.stop()
            raise

    def execute_command(self, command: str):
        """Executes a command in the persistent shell session."""
        if not self.container or not self.socket:
            raise Exception("Sandbox is not running or session is not started.")

        self.command_id += 1
        id = self.command_id
        stdout_file = f"/tmp/stdout_{id}.txt"
        stderr_file = f"/tmp/stderr_{id}.txt"
        exitcode_file = f"/tmp/exitcode_{id}.txt"
        marker = f"COMMAND_DONE_{id}"

        # By wrapping the command in `{ ...; }`, we create a command group.
        # This allows shell redirections inside the `command` (e.g., `> file.txt`)
        # to function correctly, while we capture the output of the group itself.
        # Unlike a subshell `(...)`, this executes in the current shell context,
        # so commands like `cd` work as expected.
        # The spaces and trailing semicolon are required syntax for the shell group.
        grouped_command = f"{{ {command}; }}"

        # Send command with output redirection and marker
        cmd_to_send = (
            f"{grouped_command} > {stdout_file} 2> {stderr_file}; "
            f"echo $? > {exitcode_file}; echo '{marker}'\n"
        )
        self.socket._sock.send(cmd_to_send.encode('utf-8'))

        # Wait for command completion
        self.read_until_marker(marker)

        # Read output files
        stdout_exit, (stdout_data, stdout_errdata) = self.container.exec_run(f"cat {stdout_file}", demux=True)
        stderr_exit, (stderr_data, stderr_errdata) = self.container.exec_run(f"cat {stderr_file}", demux=True)
        exitcode_exit, (exitcode_data, exitcode_errdata) = self.container.exec_run(f"cat {exitcode_file}", demux=True)
        
        if stdout_exit != 0:
            raise Exception(f"Command failed: {stdout_errdata.decode('utf-8')}")
        if stderr_exit != 0:
            raise Exception(f"Command failed: {stderr_errdata.decode('utf-8')}")
        if exitcode_exit != 0:
            raise Exception(f"Command failed: {exitcode_errdata.decode('utf-8')}")

        # Decode outputs
        stdout = stdout_data.decode('utf-8') if stdout_data else ""
        stderr = stderr_data.decode('utf-8') if stderr_data else ""
        exit_code_str = exitcode_data.decode('utf-8').strip() if exitcode_data else "0"

        # Parse exit code
        try:
            exit_code = int(exit_code_str)
        except ValueError:
            exit_code = -1  # Indicate parsing error

        # Clean up temporary files
        self.container.exec_run(f"rm {stdout_file} {stderr_file} {exitcode_file}")

        return stdout, stderr, exit_code

    def read_until_marker(self, marker, timeout=20):
        """Reads from the socket until the specified marker is found."""
        if self.socket is None:
            raise Exception("Socket not initialized")
        accumulated = ""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                data = self.socket._sock.recv(1024).decode('utf-8')
                accumulated += data
                if marker in accumulated:
                    return accumulated
            except socket.timeout:
                continue
        raise TimeoutError(f"Timeout waiting for marker: {marker}")

    def stop(self):
        """Stops the shell session and removes the container."""
        if self.socket:
            try:
                self.socket._sock.send(b"exit\n")
                time.sleep(1)  # Allow bash to exit
            except Exception as e:
                print(f"Error sending exit command: {e}")
            self.socket.close()
            self.socket = None
        if self.container:
            try:
                self.container.remove(force=True)
            except docker.errors.APIError as e: # type: ignore
                print(f"Warning: Could not stop container properly: {e}")
            self.container = None


if __name__ == "__main__":
    # Example usage
    sandbox = Sandbox()
    sandbox.start()

    # Test state persistence
    stdout, stderr, exit_code = sandbox.execute_command("mkdir /workspace")
    print(f"mkdir: Exit code: {exit_code}, Stdout: {stdout}, Stderr: {stderr}")

    stdout, stderr, exit_code = sandbox.execute_command("cd /workspace")
    print(f"cd: Exit code: {exit_code}, Stdout: {stdout}, Stderr: {stderr}")

    stdout, stderr, exit_code = sandbox.execute_command("touch file.txt")
    print(f"touch: Exit code: {exit_code}, Stdout: {stdout}, Stderr: {stderr}")

    stdout, stderr, exit_code = sandbox.execute_command("ls -l")
    print(f"ls: Exit code: {exit_code}, Stdout: {stdout}, Stderr: {stderr}")

    stdout, stderr, exit_code = sandbox.execute_command("asdf") # should fail
    print(f"cwd: Exit code: {exit_code}, Stdout: {stdout}, Stderr: {stderr}")

    sandbox.stop()
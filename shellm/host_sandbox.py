import subprocess
import time
import threading
import queue
import os
import tempfile
import uuid

class HostSandbox:
    """Manages a persistent shell session on the host machine."""
    
    def __init__(self, shell="/bin/bash", setup_commands=[]):
        self.shell = shell
        self.process = None
        self.command_id = 0
        self.setup_commands = setup_commands
        self.stdout_queue = queue.Queue()
        self.stderr_queue = queue.Queue()
        self.stdout_thread = None
        self.stderr_thread = None
        self._temp_dir = None

    def start(self):
        """Starts a new shell process and sets up a persistent session."""
        print("Starting host sandbox...")
        try:
            # Create a temporary directory for communication files
            self._temp_dir = tempfile.mkdtemp(prefix="host_sandbox_")
            
            # Start shell process with pipes
            self.process = subprocess.Popen(
                [self.shell],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,  # Unbuffered
                preexec_fn=os.setsid if os.name != 'nt' else None  # Create new process group
            )
            
            # Start threads to read stdout and stderr
            self.stdout_thread = threading.Thread(
                target=self._read_stream, 
                args=(self.process.stdout, self.stdout_queue),
                daemon=True
            )
            self.stderr_thread = threading.Thread(
                target=self._read_stream, 
                args=(self.process.stderr, self.stderr_queue),
                daemon=True
            )
            
            self.stdout_thread.start()
            self.stderr_thread.start()
            
            # Execute setup commands if provided
            if self.setup_commands:
                print("Running setup commands...")
                for cmd in self.setup_commands:
                    stdout, stderr, exit_code = self.execute_command(cmd)
                    if exit_code != 0:
                        raise Exception(f"Setup command failed: {cmd}\nStderr: {stderr}")
            
            print("Host sandbox ready.")
            
        except Exception as e:
            print(f"Error starting host sandbox: {e}")
            self.stop()
            raise

    def _read_stream(self, stream, queue_obj):
        """Reads from a stream and puts data into a queue."""
        try:
            while True:
                char = stream.read(1)
                if not char:
                    break
                queue_obj.put(char)
        except:
            pass  # Stream closed

    def execute_command(self, command: str):
        """Executes a command in the persistent shell session."""
        if not self.process or self.process.poll() is not None:
            raise Exception("Host sandbox is not running.")

        self.command_id += 1
        command_id = self.command_id
        
        # Create unique marker for this command
        marker = f"HOST_COMMAND_DONE_{command_id}_{uuid.uuid4().hex[:8]}"
        
        # Create temporary files for output capture
        stdout_file = os.path.join(self._temp_dir, f"stdout_{command_id}.txt")
        stderr_file = os.path.join(self._temp_dir, f"stderr_{command_id}.txt")
        exitcode_file = os.path.join(self._temp_dir, f"exitcode_{command_id}.txt")
        
        # Wrap command in a group to handle redirections properly
        grouped_command = f"{{ {command}; }}"
        
        # Build the full command with output redirection and marker
        full_command = (
            f"{grouped_command} > '{stdout_file}' 2> '{stderr_file}'; "
            f"echo $? > '{exitcode_file}'; "
            f"echo '{marker}'\n"
        )
        
        # Clear queues before sending command
        self._clear_queues()
        
        # Send command to shell
        try:
            self.process.stdin.write(full_command) # type: ignore
            self.process.stdin.flush() # type: ignore
        except BrokenPipeError:
            raise Exception("Shell process has terminated unexpectedly")
        
        # Wait for command completion by looking for marker
        self._wait_for_marker(marker, timeout=30)
        
        # Read the output files
        try:
            with open(stdout_file, 'r') as f:
                stdout = f.read()
        except FileNotFoundError:
            stdout = ""
        
        try:
            with open(stderr_file, 'r') as f:
                stderr = f.read()
        except FileNotFoundError:
            stderr = ""
        
        try:
            with open(exitcode_file, 'r') as f:
                exit_code_str = f.read().strip()
                exit_code = int(exit_code_str) if exit_code_str else -1
        except (FileNotFoundError, ValueError):
            exit_code = -1
        
        # Clean up temporary files
        for file_path in [stdout_file, stderr_file, exitcode_file]:
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass
        
        return stdout, stderr, exit_code

    def _clear_queues(self):
        """Clear stdout and stderr queues."""
        while not self.stdout_queue.empty():
            try:
                self.stdout_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.stderr_queue.empty():
            try:
                self.stderr_queue.get_nowait()
            except queue.Empty:
                break

    def _wait_for_marker(self, marker, timeout=30):
        """Wait for the command completion marker in stdout."""
        accumulated = ""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Read from stdout queue
                char = self.stdout_queue.get(timeout=0.1)
                accumulated += char
                
                # Check if we found the marker
                if marker in accumulated:
                    return accumulated
                    
            except queue.Empty:
                # Check if process is still alive
                if self.process.poll() is not None:
                    raise Exception("Shell process terminated unexpectedly")
                continue
        
        raise TimeoutError(f"Timeout waiting for command completion marker: {marker}")

    def stop(self):
        """Stops the shell session and cleans up resources."""
        if self.process:
            try:
                # Send exit command to gracefully close shell
                if self.process.poll() is None:
                    self.process.stdin.write("exit\n")
                    self.process.stdin.flush()
                    
                    # Wait a bit for graceful shutdown
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        # Force terminate if it doesn't exit gracefully
                        if os.name != 'nt':
                            os.killpg(os.getpgid(self.process.pid), 9)
                        else:
                            self.process.terminate()
                        self.process.wait()
                        
            except Exception as e:
                print(f"Error during graceful shutdown: {e}")
                # Force kill
                try:
                    if os.name != 'nt':
                        os.killpg(os.getpgid(self.process.pid), 9)
                    else:
                        self.process.kill()
                except:
                    pass
            
            self.process = None
        
        # Clean up temporary directory
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                import shutil
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")
            self._temp_dir = None
        
        print("Host sandbox stopped.")


if __name__ == "__main__":
    # Example usage
    sandbox = HostSandbox()
    sandbox.start()

    # Test state persistence
    stdout, stderr, exit_code = sandbox.execute_command("mkdir -p /tmp/test_workspace")
    print(f"mkdir: Exit code: {exit_code}, Stdout: '{stdout}', Stderr: '{stderr}'")

    stdout, stderr, exit_code = sandbox.execute_command("cd /tmp/test_workspace")
    print(f"cd: Exit code: {exit_code}, Stdout: '{stdout}', Stderr: '{stderr}'")

    stdout, stderr, exit_code = sandbox.execute_command("touch file.txt")
    print(f"touch: Exit code: {exit_code}, Stdout: '{stdout}', Stderr: '{stderr}'")

    stdout, stderr, exit_code = sandbox.execute_command("ls -l")
    print(f"ls: Exit code: {exit_code}, Stdout: '{stdout}', Stderr: '{stderr}'")

    stdout, stderr, exit_code = sandbox.execute_command("pwd")
    print(f"pwd: Exit code: {exit_code}, Stdout: '{stdout}', Stderr: '{stderr}'")

    stdout, stderr, exit_code = sandbox.execute_command("nonexistent_command")  # should fail
    print(f"bad command: Exit code: {exit_code}, Stdout: '{stdout}', Stderr: '{stderr}'")

    sandbox.stop() 
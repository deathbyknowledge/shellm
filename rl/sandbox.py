import asyncio
import time
import uuid
from typing import List, Tuple

import aiodocker

# TODO: Maybe this should be a system service and only use a client from
# the training process.

class Sandbox:
    """Async version of Sandbox using aiodocker for Docker container management with persistent shell."""

    def __init__(self, image: str = "shellm-sandbox:latest", setup_commands: List[str] = []):
        self.image = image
        self.setup_commands = " && ".join(setup_commands).replace("'", "'\\''") if setup_commands else ""
        self.docker = None
        self.container = None
        self.stream = None
        self.command_id = 0
        self.id = str(uuid.uuid4())  # Unique ID for reference

    def __del__(self):
      try:
        if self.docker:
          _ = self.docker.close()
        if self.container:
          _ = self.container.delete(force=True)
        if self.stream:
          _ = self.stream.close()
      except Exception as e:
        print(f"Error in __del__: {e}")

    async def start(self):
        """Async start: Creates, starts container, runs setup, attaches streams."""
        try:
            self.docker = aiodocker.Docker()
            config = {
                'Image': self.image,
                'Cmd': ['/bin/bash'],
                'Tty': True,
                'OpenStdin': True,
                'AttachStdin': True,
                'AttachStdout': True,
                'AttachStderr': True,
            }
            self.container = await self.docker.containers.create(config)
            await self.container.start()

            # Run setup if any
            if self.setup_commands:
                exec_obj = await self.container.exec(
                    cmd=f"/bin/bash -c '{self.setup_commands}'",
                    stdout=True,
                    stderr=True,
                    stdin=False,
                )
                stream = exec_obj.start(detach=False)
                stdout_bytes = b''
                stderr_bytes = b''
                
                while True:
                  output = await stream.read_out()
                  if output is None:
                    break
                  if output.stream == 1:
                    stdout_bytes += output.data
                  elif output.stream == 2:
                    stderr_bytes += output.data
                
                inspect_data = await exec_obj.inspect()
                exit_code = inspect_data['ExitCode']

                if exit_code != 0:
                    raise Exception(f"Setup failed: {stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else 'Unknown error'}")

            # Attach for persistent shell
            self.stream = self.container.attach(
                stdin=True,
                stdout=True,
                stderr=True,
                logs=False
            )

            await self.stream.write_in(b'stty -echo\n')
            await self._drain_stream()

        except Exception as e:
            await self.stop()
            raise e

    async def _drain_stream(self, timeout: float = 0.5) -> str:
        """Drain pending output from the stream until no more data for 'timeout' seconds."""
        if self.stream is None:
            raise Exception("Stream not initialized")
        drained = b''
        while True:
            try:
                output = await asyncio.wait_for(self.stream.read_out(), timeout=timeout)
                if output is None:
                    break
                drained += output.data
            except asyncio.TimeoutError:
                break
        return drained.decode('utf-8', errors='replace')
    
    async def exec_standalone_cmd(self, cmd: str) -> Tuple[str, str, int]:
      """ Executes a command in the sandbox (outside the agent session). Returns the stdout, stderr, and exit code. """
      if self.container is None:
        raise Exception("Stream not initialized")
      exec_obj = await self.container.exec(
      cmd=cmd,
        stdout=True,
        stderr=True,
        stdin=False  # Explicitly no stdin needed for cat
      )
      stream = exec_obj.start(detach=False)
      stdout_bytes = b''
      stderr_bytes = b''
      while True:
        output = await asyncio.wait_for(stream.read_out(), timeout=10)
        if output is None:
          break
        if output.stream == 1:  # Stdout
          stdout_bytes += output.data
        elif output.stream == 2:  # Stderr
          stderr_bytes += output.data
      inspect_data = await exec_obj.inspect()
      exit_code = inspect_data['ExitCode']
      return stdout_bytes.decode('utf-8', errors='replace'), stderr_bytes.decode('utf-8', errors='replace'), exit_code
    
    async def read_file_via_exec(self, file_path: str) -> str:
      stdout, stderr, exit_code = await self.exec_standalone_cmd(f"cat {file_path}")
      if exit_code != 0:
        raise RuntimeError(
          f"Failed to read {file_path}, exit code {exit_code}, "
          f"stderr: {stderr}"
        )
      return stdout


    async def exec_session_cmd(self, command: str) -> Tuple[str, str, int]:
        """Async executes a command in the persistent shell session."""
        if not self.container or not self.stream:
            raise Exception("Sandbox is not running.")

        self.command_id += 1
        if command.strip()[0] == "#":
          return "", "", 0
        id_ = self.command_id
        stdout_file = f"/tmp/stdout_{id_}.txt"
        stderr_file = f"/tmp/stderr_{id_}.txt"
        exitcode_file = f"/tmp/exitcode_{id_}.txt"
        marker = f"COMMAND_DONE_{id_}"

        grouped_command = f"{{ {command} ; }}"
        cmd_to_send = (
            f"{grouped_command} > {stdout_file} 2> {stderr_file}; "
            f"echo $? > {exitcode_file}; echo '{marker}'\n"
        )

        await self.stream.write_in(cmd_to_send.encode('utf-8'))

        # Wait for completion
        await self.read_until_marker(marker)

        # Read files using exec
        stdout_bytes = await self.read_file_via_exec(stdout_file)
        stderr_bytes = await self.read_file_via_exec(stderr_file)
        exitcode_bytes = await self.read_file_via_exec(exitcode_file)

        stdout = stdout_bytes if stdout_bytes else ""
        stderr = stderr_bytes if stderr_bytes else ""
        exit_code_str = exitcode_bytes if exitcode_bytes else "0"

        try:
            exit_code = int(exit_code_str)
        except ValueError:
            exit_code = -1

        # Clean up files
        exec_obj = await self.container.exec(cmd=f"rm {stdout_file} {stderr_file} {exitcode_file}", stdout=False, stderr=False, stdin=False)
        _ = await exec_obj.start(detach=True)  # Ignore output/exit code

        await self._drain_stream()

        return stdout, stderr, exit_code

    async def read_until_marker(self, marker: str, timeout: float = 10.0):
        """Async reads from the reader stream until the marker is found."""
        if self.stream is None:
            raise Exception("Stream not initialized")
        accumulated = ""
        start_time = time.time()
        while True:
            output = await asyncio.wait_for(self.stream.read_out(), timeout=timeout)
            if output is None:
              raise Exception("Stream closed unexpectedly")
            chunk = output.data.decode('utf-8', errors='replace') # TODO: look into errors=replace
            accumulated += chunk
            if marker in accumulated:
                return
            if (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Timeout waiting for marker: {marker}")

    async def stop(self):
        """Async stops the stream and removes the container."""
        if self.stream:
            await self.stream.write_in(b"exit\n")
            await asyncio.sleep(1)
            await self.stream.close()
            self.stream = None
        if self.container:
            await self.container.delete(force=True)
            self.container = None
        if self.docker:
            await self.docker.close()
            self.docker = None
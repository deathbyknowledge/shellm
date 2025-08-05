import asyncio
import httpx

TIMEOUT = 300

class SoSClient:
  def __init__(self, server_url="http://localhost:3000"):
    self.server_url = server_url

  async def create_sandbox(self, image="ubuntu:latest", setup_commands=None):
      """
      Creates a new sandbox.

      Args:
          image (str): The container image to use.
          setup_commands (list): A list of commands to run after the container starts.
          server_url (str): The base URL of the sandbox server.

      Returns:
          dict: The JSON response from the server, typically containing the new sandbox ID.
      """
      if setup_commands is None:
          setup_commands = []
      payload = {"image": image, "setup_commands": setup_commands}
      async with httpx.AsyncClient(timeout=TIMEOUT) as client:
          response = await client.post(f"{self.server_url}/sandboxes", json=payload)
          response.raise_for_status()
          parsed = response.json()
          if "id" not in parsed:
            raise Exception(f"Failed to create sandbox: {parsed}")
          return parsed["id"]

  async def list_sandboxes(self):
      """
      Lists all available sandboxes.

      Args:
          server_url (str): The base URL of the sandbox server.

      Returns:
          list: A list of dictionaries, each containing information about a sandbox.
      """
      async with httpx.AsyncClient(timeout=TIMEOUT) as client:
          response = await client.get(f"{self.server_url}/sandboxes")
          response.raise_for_status()
          return response.json()

  async def start_sandbox(self, sandbox_id):
      """
      Starts a specific sandbox.

      Args:
          sandbox_id (str): The ID of the sandbox to start.
          server_url (str): The base URL of the sandbox server.
      """
      async with httpx.AsyncClient(timeout=TIMEOUT) as client:
          response = await client.post(f"{self.server_url}/sandboxes/{sandbox_id}/start")
          response.raise_for_status()

  async def exec_command(self, sandbox_id, command, standalone=False):
      """
      Executes a command in a specific sandbox.

      Args:
          sandbox_id (str): The ID of the sandbox to execute the command in.
          command (str): The command to execute.
          server_url (str): The base URL of the sandbox server.

      Returns:
          dict: The JSON response from the server, containing stdout, stderr, and exit code.
      """
      payload = {"command": command, "standalone": standalone}
      async with httpx.AsyncClient(timeout=TIMEOUT) as client:
          response = await client.post(f"{self.server_url}/sandboxes/{sandbox_id}/exec", json=payload)
          if response.status_code > 400:
            raise Exception(f"Failed to execute command `{command}`: {response.text}")
          parsed = response.json()
          if "output" not in parsed or "exit_code" not in parsed:
            raise Exception(f"Failed to execute command: {parsed}")
          return parsed["output"], parsed["exit_code"]

  async def stop_sandbox(self, sandbox_id, remove=True):
      """
      Stops and removes a specific sandbox.

      Args:
          sandbox_id (str): The ID of the sandbox to stop.
          server_url (str): The base URL of the sandbox server.
      """
      async with httpx.AsyncClient(timeout=TIMEOUT) as client:
          response = await client.post(f"{self.server_url}/sandboxes/{sandbox_id}/stop", json={'remove': remove})
          response.raise_for_status()

async def main():
    # Example workflow
    sandbox_id = None
    try:
        # Create a sandbox
        client = SoSClient(server_url="http://rearden:3000")
        sandbox_id = await client.create_sandbox(setup_commands=["echo 'Hello, World!' > /tmp/hello.txt"])

        # Start the sandbox
        await client.start_sandbox(sandbox_id)

        # Execute a command
        exec_response = await client.exec_command(sandbox_id, "cat /tmp/hello.txt")

        # List sandboxes
        sandboxes = await client.list_sandboxes()

    finally:
        # Stop the sandbox
        if sandbox_id is not None:
            await client.stop_sandbox(sandbox_id)

if __name__ == '__main__':
    asyncio.run(main())

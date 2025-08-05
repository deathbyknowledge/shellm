import os
import art
import json
import traceback
from pydantic import BaseModel
from openai import AsyncOpenAI
from rich import print
from typing import List, Tuple
from dotenv import load_dotenv
from json_repair import repair_json
from tenacity import retry, stop_after_attempt

load_dotenv()


from project_types import Scenario, Message
from sandbox import SoSClient

LOCAL = os.getenv("LOCAL", "1") == "1"
EPHEMERAL = os.getenv("EPHEMERAL", "1") == "1"
MAX_TURNS = int(os.getenv("MAX_TURNS", "30")) # reasoning counts as 1 turn 
MAX_MODEL_TOKENS = int(os.getenv("MAX_MODEL_TOKENS", "32000"))
BASE_URL = os.getenv("BASE_URL", "http://rearden:8000/v1")
API_KEY = os.getenv("API_KEY", "MEOW")
oai = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
sos = SoSClient(server_url="http://localhost:3000")

class ProjectTrajectory(art.Trajectory):
  task_id: str
  sandbox_id: str
  exit_codes: List[int]
  success_condition_passed: bool
  corrupted: bool

  def format_trajectory(self):
    messages = self.messages()
    if not messages or len(messages) < 2:
        return "No actions were taken."

    formatted = ""
    outputs = 0
    for msg in messages:
        role = msg['role']
        content = msg['content'] if 'content' in msg else "None"
        if role == 'assistant':
          # as command
          formatted += f"$ {content}\n"
        elif role == 'user':
          # as output
          formatted += f"{content}\n[EXIT_CODE = {self.exit_codes[outputs]}]\n"
          outputs += 1
        elif role == 'system':
          # as task (should be the first message)
          formatted += f"TASK: {content}\n\n"
    return formatted


async def run_agent(model: art.Model, scenario: Scenario) -> ProjectTrajectory:
  client = model.openai_client() if LOCAL else oai
  sandbox_id = await sos.create_sandbox(image="shellm-sandbox:latest", setup_commands=scenario.setup_commands)
  traj = ProjectTrajectory(
    reward=0.0,
    messages_and_choices=[],
    task_id=scenario.id,
    sandbox_id=sandbox_id,
    exit_codes=[],
    success_condition_passed=False,
    corrupted=False,
  )

  system_prompt = scenario.task

  traj.messages_and_choices = [
    {"role": "system", "content": system_prompt }
  ]
  traj.exit_codes = []

  try:
    await sos.start_sandbox(sandbox_id)
  except Exception as e: 
    print(scenario.setup_commands)
    raise e

  async def finish_traj(sandbox_id: str, success_command: str) -> bool:
    try:
      _, code = await sos.exec_command(sandbox_id, success_command, standalone=True)
      await sos.stop_sandbox(sandbox_id, remove=EPHEMERAL)
      return code == 0
    except Exception as e:
      await sos.stop_sandbox(sandbox_id, remove=EPHEMERAL)
      print(f"[ {scenario.id} ] Error running success command in sandbox: {e}")
      return False

  for turn in range(MAX_TURNS):

    @retry(stop=stop_after_attempt(3))
    async def get_response():
      response = await client.chat.completions.create(
        messages=traj.messages(),
        model=model.name,
        temperature=0.7,
        top_p=0.95,
        # extra_body={
        #   "top_k":50,
        # }
      )

      if not response.choices[0].message.content or response.choices[0].message.content is None:
        raise Exception("No response from model")

      return response.choices[0]
    
    approx_token_count = 0
    for msg in traj.messages():
      approx_token_count += (len(msg['content']) / 4)
    if approx_token_count > MAX_MODEL_TOKENS:
      await finish_traj(sandbox_id, scenario.success_condition)
      traj.success_condition_passed = False
      return traj

    response_message = await get_response()

    traj.messages_and_choices.append(
      response_message
    )
    
    cmd = response_message.message.content
  
    try:
      output, exit_code = await sos.exec_command(sandbox_id, cmd) 

      traj.messages_and_choices.append(
        {"role":"user", "content": output}
      )


      if "exit" in cmd and not cmd.startswith("#"):
        # It's over
        if len(cmd) > len("exit 0"):
          print(f"Exit cmd: {cmd}")
        break

      if exit_code != -1:
        traj.exit_codes.append(exit_code)
      else:
        print("-1 exit code detected")
    
    except Exception as e:
      print(f"Error running command in sandbox: {e}")
      traceback.print_exc()
      output = f"Error running command: {e}"
      traj.messages_and_choices.append({"role": "user", "content": output})
      traj.exit_codes.append(-1)
      await finish_traj(sandbox_id, scenario.success_condition)
      traj.success_condition_passed = False
      traj.corrupted = False
      return traj

  condition_passed = await finish_traj(sandbox_id, scenario.success_condition)
  traj.success_condition_passed = condition_passed
  return traj

async def run_agent_and_score(
  model: art.Model, scenario: Scenario
) -> ProjectTrajectory:
  traj = await run_agent(model, scenario)
  
  def check_exit_codes(exit_codes: List[int]) -> float:
    r = sum([-0.1 for x in exit_codes if x != 0]) 
    return r

  def check_success_command(passed: bool) -> float:
    r = 1.0 if passed else 0.0
    return r

  def check_format(messages: List[Message]) -> float:
    r = sum([-0.1 for msg in messages if "<think>" in msg['content'] or "</think>" in msg['content']]) # type: ignore
    return r

  def check_turns(messages: List[Message]) -> float:
    # The less turns, the better
    r = -0.2 if len(messages) > 15 else 0.0
    return r
    

  reward = 0.0
  # LLM Judge assigned reward
  if not traj.corrupted:
    try:
      reward += check_success_command(traj.success_condition_passed)
      if reward > 0.0:
        extra_reward = check_exit_codes(traj.exit_codes)
        extra_reward += check_format(traj.messages()) # type: ignore
        extra_reward += check_turns(traj.messages()) # type: ignore
        reward += extra_reward if extra_reward > -0.5 else -0.5
    except Exception as e:
      traj.corrupted = True
  traj.reward = reward
  return traj


if __name__ == "__main__":
  import asyncio
  from load_scenarios import load_scenarios

  scenario = load_scenarios(limit=1)[0]
  model = art.Model(name="deathbyknowledge/AFM-4.5B-Shell-SFT", project="shell-agent-test")
  traj = asyncio.run(run_agent_and_score(model, scenario))
  print(traj.format_trajectory())
  print(traj.reward)

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
from sandbox import Sandbox, SoSClient

LOCAL = False
MAX_TURNS = 30 # reasoning counts as 1 turn 
BASE_URL = "http://rearden:8000/v1"
API_KEY = "MEOW"
oai = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
JUDGE_BASE_URL = "https://api.deepseek.com"
JUDGE_API_KEY = os.getenv("DEEPSEEK_API_KEY")
JUDGE_MODEL = "deepseek-chat"
judge_oai = AsyncOpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY)
sos = SoSClient(server_url="http://localhost:3000")

DEFAULT_JUDGE_PROMPT = """
You are an expert evaluator. Your role is to determine if a task, executed in a Linux shell environment, was successfully completed based on the provided trajectory.
The agent MUST always provide reasoning for their actions by submitting bash comments. This is mandatory and does not discount the agent's score.
You will be given the initial task, any setup commands that were run, and the bash history of the agent. We have included the exit codes of the commands in the history.

Rate the trajectory on a scale from 1 to 5, where:
1 = Completely incorrect. The agent made no progress or took actions completely unrelated to the task.
2 = Mostly incorrect. The agent took some initial steps that were related to the task but failed to make substantial progress or went down a wrong path.
3 = Partially correct. The agent completed a significant portion of the task but had major errors or failed to complete all requirements.
4 = Mostly correct. The agent successfully completed the task but took unnecessary steps, had minor, recoverable errors, or the solution was inefficient.
5 = Perfectly correct. The agent completed the task efficiently and accurately, following best practices.

Provide a concise reasoning for your decision and then the final numeric rating.
Respond with a single JSON object containing 'reasoning' and 'rating'  keys.
"""

@retry(stop=stop_after_attempt(3))
async def judge_correctness(
  scenario: Scenario, messages: List[Message], exit_codes: List[int]
) -> float:
  task = scenario.task
  setup_commands = scenario.setup_commands
  def format_history(messages: List[Message], exit_codes: List[int]):
    """Formats the trajectory into a string for the prompt."""
    if not messages or len(messages) < 2:
        return "No actions were taken."

    formatted = ""
    if messages[0]['role'] == 'system': # type: ignore
      messages = messages[1:]
    commands = [msg['content'] for msg in messages if msg['role'] == 'assistant'] # type: ignore
    outputs = [msg['content'] for msg in messages if msg['role'] == 'user'] # type: ignore
    for (turn, command) in enumerate(commands):
        if "exit 0" in command:
          break
        formatted += f"$ {command}\n"
        formatted += f"{outputs[turn].strip()}\n[EXIT_CODE = {exit_codes[turn]}]\n"
    return formatted

  history = format_history(messages, exit_codes)
  setup_str = "\n".join(f"$ {cmd}" for cmd in setup_commands) if setup_commands else "None"

  prompt = f"TASK: {task}\n\nSETUP COMMANDS:\n{setup_str}\n\nTRAJECTORY:\n{history}\n\nBased on the trajectory, was the task successfully completed? Provide your reasoning and rating."
  judge_response = await judge_oai.chat.completions.create(
      model=JUDGE_MODEL,
      messages=[
          {"role": "system", "content": DEFAULT_JUDGE_PROMPT},
          {"role": "user", "content": prompt}
      ],
      max_tokens=2048,
      temperature=0.0,
  )
  judge_response = json.loads(repair_json(str(judge_response.choices[0].message.content)))
  reasoning = judge_response['reasoning']
  rating = judge_response['rating']
  reward = 0.0
  # TODO: don't use yolo numbers
  if rating == 5:
      reward = 1.0
  elif rating == 4:
      reward = 0.5
  elif rating == 3:
      reward = 0.0
  elif rating == 2:
      reward = -0.5
  elif rating == 1:
      reward = -1.0
  return reward

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
    except Exception as e:
      print(f"[ {scenario.id} ] Error running success command in sandbox: {e}")
    finally:
      await sos.stop_sandbox(sandbox_id)
      return code == 0

  for turn in range(MAX_TURNS):

    @retry(stop=stop_after_attempt(3))
    async def get_response():
      response = await client.chat.completions.create(
        messages=traj.messages(),
        model=model.name,
      )

      return response.choices[0]
    
    response_message = await get_response()

    traj.messages_and_choices.append(
      response_message
    )

    if not response_message.message.content or response_message.message.content is None:
      # We always expect text content. If it's missing, the model isn't
      # behaving as we want it to and we return the trajectory.
      condition_passed = await finish_traj(sandbox_id, scenario.success_condition)
      traj.success_condition_passed = condition_passed
      traj.corrupted = True
      return traj
    
    cmd = response_message.message.content
  
    if "exit 0" in cmd:
      # It's over
      condition_passed = await finish_traj(sandbox_id, scenario.success_condition)
      traj.success_condition_passed = condition_passed
      return traj

    try:
      output, exit_code = await sos.exec_command(sandbox_id, cmd) 

      traj.messages_and_choices.append(
        {"role":"user", "content": output}
      )

      if exit_code != -1:
        traj.exit_codes.append(exit_code)
    
    except Exception as e:
      print(f"Error running command in sandbox: {e}")
      traceback.print_exc()
      output = f"Error running command: {e}"
      traj.messages_and_choices.append({"role": "user", "content": output})
      traj.exit_codes.append(-1)
      traj.success_condition_passed = False
      await finish_traj(sandbox_id, scenario.success_condition)
      traj.corrupted = True
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
      # reward += await judge_correctness(
      #   scenario, traj.messages(),traj.exit_codes # type: ignore
      # )
      # Reward discounts based off how many error exit codes
      # there were in the trajectory
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
  model = art.Model(name="deathbyknowledge/Qwen2.5-7B-Shell-SFT", project="shell-agent-test")
  traj = asyncio.run(run_agent_and_score(model, scenario))
  print(traj.format_trajectory())
  print(traj.reward)

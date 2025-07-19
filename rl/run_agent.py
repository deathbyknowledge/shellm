import os
import art
import json
from pydantic import BaseModel
from openai import AsyncOpenAI
from rich import print
from typing import List, Tuple
from dotenv import load_dotenv
from json_repair import repair_json

load_dotenv()


from project_types import Scenario, Message
from sandbox import Sandbox

LOCAL = True
MAX_TURNS = 30 # reasoning counts as 1 turn 
BASE_URL = "http://rearden:8000/v1"
API_KEY = "MEOW"
oai = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
JUDGE_BASE_URL = "https://api.deepseek.com"
JUDGE_API_KEY = os.getenv("DEEPSEEK_API_KEY")
JUDGE_MODEL = "deepseek-chat"
judge_oai = AsyncOpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY)

DEFAULT_JUDGE_PROMPT = """
You are an expert evaluator. Your role is to determine if a task, executed in a Linux shell environment, was successfully completed based on the provided trajectory.
The agent MUST always provide reasoning for their actions by submitting bash comments. This is mandatory and does not discount the agent's score.
The agent must also use `exit 0` once the task is completed.
You will be given the initial task, any setup commands that were run, and the full sequence of shell interactions, and outputs from the AI agent.

Rate the trajectory on a scale from 1 to 5, where:
1 = Completely incorrect. The agent made no progress or took actions completely unrelated to the task.
2 = Mostly incorrect. The agent took some initial steps that were related to the task but failed to make substantial progress or went down a wrong path.
3 = Partially correct. The agent completed a significant portion of the task but had major errors or failed to complete all requirements.
4 = Mostly correct. The agent successfully completed the task but took unnecessary steps, had minor, recoverable errors, or the solution was inefficient.
5 = Perfectly correct. The agent completed the task efficiently and accurately, following best practices.

Provide a concise reasoning for your decision and then the final numeric rating.
Respond with a single JSON object containing 'reasoning' and 'rating'  keys.
"""

async def judge_correctness(
  scenario: Scenario, messages: List[Message], exit_codes: List[int]
) -> float:
  task = scenario.task
  setup_commands = scenario.setup_commands
  def format_history(messages: List[Message], exit_codes: List[int]):
    """Formats the trajectory into a string for the prompt."""
    if not messages or len(messages) < 2:
        return "No actions were taken."

    formatted = []
    if messages[0]['role'] == 'system': # type: ignore
      messages = messages[1:]
    commands = [msg['content'] for msg in messages if msg['role'] == 'assistant'] # type: ignore
    outputs = [msg['content'] for msg in messages if msg['role'] == 'user'] # type: ignore
    for (turn, command) in enumerate(commands):
        formatted.append(f"Turn {turn}:")
        formatted.append(f"Action:`{command}`") # type: ignore
        if "exit 0" in command:
          break
        formatted.append(f"Exit Code: {exit_codes[turn]}")
        formatted.append(f"Shell Output:\n---\n{outputs[turn].strip()}\n---") # type: ignore
    return "\n".join(formatted)

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
  )
  judge_response = json.loads(repair_json(str(judge_response.choices[0].message.content)))
  reasoning = judge_response['reasoning']
  rating = judge_response['rating']
  reward = 0.0
  # TODO: don't use yolo numbers
  if rating == 5:
      reward = 1.0
  elif rating == 4:
      reward = 0.4
  elif rating == 3:
      reward = 0.0
  elif rating == 2:
      reward = -0.2
  elif rating == 1:
      reward = -0.5
  print(f"Judge reward: {reward}")
  return reward

class ProjectTrajectory(art.Trajectory):
  exit_codes: List[int]
  success_condition_passed: bool


async def run_agent(model: art.Model, scenario: Scenario) -> ProjectTrajectory:
  client = model.openai_client() if LOCAL else oai
  traj = ProjectTrajectory(
    reward=0.0,
    messages_and_choices=[],
    exit_codes=[],
    success_condition_passed=False,
  )

  system_prompt = scenario.task

  traj.messages_and_choices = [
    {"role": "system", "content": system_prompt }
  ]
  traj.exit_codes = []

  sandbox = Sandbox(setup_commands=scenario.setup_commands)
  await sandbox.start()

  async def finish_traj(sandbox: Sandbox, success_command: str) -> bool:
    _, _, code = await sandbox.execute_command(success_command)
    await sandbox.stop()
    return code == 0

  for turn in range(MAX_TURNS):
    response = await client.chat.completions.create(
      messages=traj.messages(),
      model=model.name,
    )

    response_message = response.choices[0]

    traj.messages_and_choices.append(
      response_message
    )

    if not response_message.message.content:
      # We always expect text content. If it's missing, the model isn't
      # behaving as we want it to and we return the trajectory.
      condition_passed = await finish_traj(sandbox, scenario.success_condition)
      traj.success_condition_passed = condition_passed
      return traj
    
    cmd = response_message.message.content
  
    if "exit 0" in cmd:
      # It's over
      condition_passed = await finish_traj(sandbox, scenario.success_condition)
      traj.success_condition_passed = condition_passed
      return traj

    try:
      stdout, stderr, exit_code = await sandbox.execute_command(cmd) 

      output = ""
      if stdout is not None:
        output += stdout
      if stderr is not None:
        output += stderr

      traj.messages_and_choices.append(
        {"role":"user", "content": output}
      )

      if exit_code != -1:
        traj.exit_codes.append(exit_code)
    
    except Exception as e:
      print(f"Error running command in sandbox: {e}")
      condition_passed = await finish_traj(sandbox, scenario.success_condition)
      traj.success_condition_passed = condition_passed
      return traj

  condition_passed = await finish_traj(sandbox, scenario.success_condition)
  traj.success_condition_passed = condition_passed
  return traj

async def run_agent_and_score(
  model: art.Model, scenario: Scenario
) -> ProjectTrajectory:
  traj = await run_agent(model, scenario)
  
  def check_exit_codes(exit_codes: List[int]) -> float:
    r = sum([-0.05 for x in exit_codes if x != 0]) 
    print(f"Exit code reward: {r}")
    return r

  def check_success_command(passed: bool) -> float:
    r = 0.2 if passed else 0.0
    print(f"Success command reward: {r}")
    return r

  def check_format(messages: List[Message]) -> float:
    r = sum([-0.05 for msg in messages if "<think>" in msg['content'] or "</think>" in msg['content']]) # type: ignore
    print(f"Format reward: {r}")
    return r
    

  reward = 0.0
  # LLM Judge assigned reward
  reward += await judge_correctness(
    scenario, traj.messages(),traj.exit_codes # type: ignore
  )
  # Reward discounts based off how many error exit codes
  # there were in the trajectory
  reward += check_exit_codes(traj.exit_codes)
  reward += check_success_command(traj.success_condition_passed)
  reward += check_format(traj.messages()) # type: ignore
  traj.reward = reward
  return traj # type: ignore


if __name__ == "__main__":
  import asyncio
  from load_scenarios import load_scenarios

  scenario = load_scenarios(limit=1)[0]
  model = art.Model(name="deathbyknowledge/Qwen3-8B-Shell-SFT", project="shell-agent-test")
  traj = asyncio.run(run_agent_and_score(model, scenario))
  print(traj.reward)

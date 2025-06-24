import os
from openai import OpenAI
from pydantic import BaseModel, Field
import instructor
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

class JudgeResponse(BaseModel):
    reasoning: str = Field(..., description="A detailed explanation for the given rating, referencing specific turns and observations.")
    rating: int = Field(..., description="The numeric rating from 1 to 5.", ge=1, le=5)

class Judge:
    """Handles the evaluation of a completed trajectory by an LLM."""

    def __init__(self, base_url: str | None = "https://api.deepseek.com", api_key: str | None = os.environ.get("DEEPSEEK_API_KEY"), model: str = "deepseek-chat"):
        oai = OpenAI(base_url=base_url, api_key=api_key)
        self.client = instructor.from_openai(oai)
        self.model = model

    def _format_history(self, trajectory: List[Dict[str, Any]]):
        """Formats the trajectory into a string for the prompt."""
        if not trajectory:
            return "No actions were taken."

        formatted = []
        for turn in trajectory:
            formatted.append(f"Turn {turn['turn']}:")
            formatted.append(f"Thought:\n{turn['thought']}")
            formatted.append(f"Action: `{turn['action']}`")
            formatted.append(f"Exit Code: {turn['exit_code']}")
            formatted.append(f"Observation:\n---\n{turn['observation'].strip()}\n---")

        return "\n".join(formatted)

    def evaluate_trajectory(self, task: str, setup_commands: List[str], trajectory: List[Dict[str, Any]]):
        """Evaluates a trajectory to determine if the task was successfully completed."""

        system_prompt = """
        You are an expert evaluator. Your role is to determine if a task, executed in a Linux shell environment, was successfully completed based on the provided trajectory.
        You will be given the initial task, any setup commands that were run, and the full sequence of thoughts, actions, and observations from an AI agent.

        Rate the trajectory on a scale from 1 to 5, where:
        1 = Completely incorrect. The agent made no progress or took actions completely unrelated to the task.
        2 = Mostly incorrect. The agent took some initial steps that were related to the task but failed to make substantial progress or went down a wrong path.
        3 = Partially correct. The agent completed a significant portion of the task but had major errors or failed to complete all requirements.
        4 = Mostly correct. The agent successfully completed the task but took unnecessary steps, had minor, recoverable errors, or the solution was inefficient.
        5 = Perfectly correct. The agent completed the task efficiently and accurately, following best practices.

        Provide a concise reasoning for your decision and then the final numeric rating.
        Respond with a single JSON object containing 'reasoning' and 'rating'  keys.
        """

        history = self._format_history(trajectory)
        setup_str = "\n".join(f"$ {cmd}" for cmd in setup_commands) if setup_commands else "None"

        prompt_content = f"TASK: {task}\n\nSETUP COMMANDS:\n{setup_str}\n\nTRAJECTORY:\n{history}\n\nBased on the trajectory, was the task successfully completed? Provide your rating and reasoning."

        try:
            judge_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_content}
                ],
                response_model=JudgeResponse,
                temperature=0.1
            )
            return judge_response
        except Exception as e:
            print(f"Error getting judgment from LLM: {e}")
            return JudgeResponse(rating=1, reasoning=f"Failed to get evaluation due to an error: {e}")

if __name__ == "__main__":
    # Example usage
    judge = Judge()
    example_task = "Create a file named 'hello.txt' containing the text 'Hello, Judge!'."
    example_setup = ["mkdir -p /app"]
    example_trajectory = [
        {
            "turn": 1,
            "thought": "I need to create a directory first, just in case. Then I'll create the file.",
            "action": "cd /app",
            "observation": "",
            "exit_code": 0
        },
        {
            "turn": 2,
            "thought": "Now I am in the /app directory. I will create the file and write the text to it.",
            "action": "echo 'Hello, Judge!' > hello.txt",
            "observation": "",
            "exit_code": 0
        },
        {
            "turn": 3,
            "thought": "I should verify that the file was created with the correct content.",
            "action": "cat hello.txt",
            "observation": "Hello, Judge!",
            "exit_code": 0
        }
    ]
    evaluation = judge.evaluate_trajectory(example_task, example_setup, example_trajectory)
    print("--- Evaluation ---")
    print(f"Rating: {evaluation.rating}")
    print(f"Reasoning: {evaluation.reasoning}")
import os
from openai import OpenAI
from pydantic import BaseModel
import instructor

class ShellResponse(BaseModel):
    thought: str
    action: str

def get_next_step(self, task, trajectory):
    """Constructs a prompt and gets the next thought/action."""
    
    system_prompt = """
    You are an expert Linux shell user. Your goal is to complete the given task step-by-step.
    Given a task and the history of previous turns, provide your reasoning in a 'thought' process
    and then provide the single, next shell command to execute in an 'action'.
    Pay attention to the exit code of the previous action. If it is not 0, you should provide a
    debugging action to fix the issue and complete the previous turn's goal.
    The 'action' should be a valid, single-line shell command.
    If the task is complete, the action should be exactly 'exit 0'.
    Respond with a single JSON object containing 'thought' and 'action' keys.
    """
    
    history = self._format_history(trajectory)
    prompt_content = f"TASK: {task}\n\nHISTORY:\n{history}\n\nProvide the next step."

    shell_response = self.client.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_content}
        ],
        response_model=ShellResponse,
        temperature=0.4
    )
    
    return shell_response.thought, shell_response.action

class Teacher:
    """Handles interaction with the teacher language model."""
    
    def __init__(self):
        oai = OpenAI(base_url="https://api.deepseek.com", api_key=os.environ.get("DEEPSEEK_API_KEY"))
        self.client = instructor.from_openai(oai)
        self.model = "deepseek-chat"

    def get_next_step(self, task, trajectory):
        """Constructs a prompt and gets the next thought/action."""
        
        system_prompt = """
        You are an expert Linux shell user. Your goal is to complete the given task step-by-step.
        Given a task and the history of previous turns, provide your reasoning in a 'thought' process
        and then provide the single, next shell command to execute in an 'action'.
        Pay attention to the exit code of the previous action. If it is not 0, you should provide a
        debugging action to fix the issue and complete the previous turn's goal.
        The 'action' should be a valid, single-line shell command.
        If the task is complete, the action should be exactly 'exit 0'.
        Respond with a single JSON object containing 'thought' and 'action' keys.
        """
        
        history = self._format_history(trajectory)
        prompt_content = f"TASK: {task}\n\nHISTORY:\n{history}\n\nProvide the next step."

        shell_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_content}
            ],
            response_model=ShellResponse,
            temperature=0.4
        )
        if not shell_response:
            raise ValueError("Received an empty response from the language model.")
        
        return shell_response.thought, shell_response.action

    def _format_history(self, trajectory):
        """Formats the trajectory into a string for the prompt."""
        if not trajectory:
            return "No history yet. This is the first step."
        
        formatted = []
        for turn in trajectory:
            formatted.append(f"Turn {turn['turn']}:")
            formatted.append(f"Thought:\n{turn['thought']}")
            formatted.append(f"Action: `{turn['action']}`")
            formatted.append(f"Exit Code: {turn['exit_code']}")
            formatted.append(f"Observation:\n---\n{turn['observation'].strip()}\n---")
        
        return "\n".join(formatted)

if __name__ == "__main__":
    teacher = Teacher()
    teacher.get_next_step("Create a new file called 'test.txt' and write 'Hello, world!' to it.", [])

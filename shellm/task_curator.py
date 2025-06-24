import random
import json

from pydantic import BaseModel


class RequirementsResponse(BaseModel):
    reasoning: str
    setup_commands: list[str]


class TaskCurator:
    """Provides a list of diverse, high-quality tasks from a text file."""
    
    def __init__(self, seed: int = 42, task_file: str = "tasks.jsonl"):
        self.seed = seed
        self.task_file = task_file
        self.tasks = self._load_tasks()

    def _load_tasks(self):
        """Load tasks from output.txt and assign them task IDs."""
        try:
            with open(self.task_file, 'r') as file:
                tasks = [json.loads(line.strip()) for line in file if line.strip()]  # Read non-empty lines
            # Shuffle tasks with the provided seed for reproducibility
            random.seed(self.seed)
            random.shuffle(tasks)
            # Add task IDs (t001, t002, etc.)
            for i, task in enumerate(tasks):
                task['id'] = f"t{str(i+1).zfill(3)}"
            return tasks
        except FileNotFoundError:
            print(f"Error: {self.task_file} not found. Returning empty task list.")
            return {}
        except Exception as e:
            print(f"Error loading tasks: {e}. Returning empty task list.")
            return {}

    def get_tasks(self, limit=5):
        """Returns a specified number of tasks."""
        return self.tasks[:limit]

if __name__ == "__main__":
    curator = TaskCurator(seed=420)
    curator.get_tasks(limit=20)




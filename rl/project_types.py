from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class Message(BaseModel):
  role: Literal['assistant', 'user', 'system']
  content: str

class Scenario(BaseModel):
  id: str
  task: str
  solution: List[Message]
  setup_commands: List[str]
  success_condition: str


class RunConfig(BaseModel):
  num_epochs: int = 1
  groups_per_step: int = 12
  validation_frequency: int = 10
  validation_num_scenarios: int = 14
  training_num_scenarios: int = 1000
  rollouts_per_group: int = 4
  learning_rate: float = 1.2e-5

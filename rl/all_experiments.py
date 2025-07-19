import art
from project_types import RunConfig


models: dict[str, art.TrainableModel[RunConfig]] = {}

models["run_1"] = art.TrainableModel[RunConfig](
    base_model="Qwen/Qwen2.5-14B-Instruct",
    project="agent-class-art",
    name="run_1",
    config=RunConfig(),
)

models["run_2"] = models["run_1"].model_copy(deep=True)
models["run_2"].name = "run_2"
models["run_2"].config.rollouts_per_group = 8
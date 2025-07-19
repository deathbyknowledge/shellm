import art
from run_agent import run_agent_and_score
from load_scenarios import load_scenarios
from art.local import LocalBackend
from art.utils import iterate_dataset
from project_types import RunConfig
from benchmark import benchmark


async def train(model: art.TrainableModel[RunConfig]):
    training_data = load_scenarios(split="train", limit=model.config.training_num_scenarios)

    with LocalBackend() as backend:
        await model.register(backend)

        training_iterator = iterate_dataset(
            training_data,
            groups_per_step=model.config.groups_per_step,
            num_epochs=model.config.num_epochs,
            initial_step=await model.get_step(),
        )

        for batch, epoch, global_step, epoch_step in training_iterator:
            if global_step % model.config.validation_frequency == 0:
                results, score = await benchmark(
                    model, model.config.validation_num_scenarios,
                )
                await model.log(results)
            groups = []
            for scenario in batch:
                groups.append(
                    art.TrajectoryGroup(
                        (
                            run_agent_and_score(model, scenario)
                            for _ in range(model.config.rollouts_per_group)
                        )
                    )
                )
            finished_groups = await art.gather_trajectory_groups(groups)
            await model.train(
                finished_groups,
                config=art.TrainConfig(learning_rate=model.config.learning_rate),
            )

if __name__ == "__main__":
    import asyncio
    from all_experiments import models
    import argparse

    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--model",
        required=True,
        help="The key of the model to train as defined in all_experiments.py (e.g. 'run_1')",
    )
    args = parser.parse_args()
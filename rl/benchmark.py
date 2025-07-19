import art
from run_agent import run_agent_and_score, ProjectTrajectory
from load_scenarios import load_scenarios
from tqdm.asyncio import tqdm

async def benchmark(
    model: art.Model, num_scenarios: int
) -> tuple[list[ProjectTrajectory], float]:
    scenarios = load_scenarios(limit=num_scenarios, split="test")
    results: list[ProjectTrajectory] = await tqdm.gather(
        *[run_agent_and_score(model, scenario) for scenario in scenarios],
        desc=f"benchmarking {model.name}",
    )
    scores = [result.reward for result in results]
    return results, sum(scores) / len(scores) if scores else 0


async def benchmark_all_models(num_scenarios: int) -> dict[str, float]:
    model_names = [
        "deathbyknowledge/Qwen3-8B-Shell-SFT",
    ]

    models = [art.Model(name=name, project="shell-agent-test") for name in model_names]
    results = await asyncio.gather(
        *[benchmark(model, num_scenarios) for model in models]
    )
    return {model.name: score for model, (_results, score) in zip(models, results)}


if __name__ == "__main__":
    import asyncio
    from rich import print

    print(asyncio.run(benchmark_all_models(num_scenarios=14)))
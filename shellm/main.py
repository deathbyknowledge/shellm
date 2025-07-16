# main.py
from tqdm import tqdm
from task_curator import TaskCurator
from teacher import Teacher, ShellTeacher
from sandbox import Sandbox
from judge import Judge

import aiofiles
from dotenv import load_dotenv
import json
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import subprocess
import requests
import argparse

load_dotenv()

def generate_trajectory(task_id, task_description, setup_commands, how_realistic, difficulty_level, required_tools, success_condition, run_evaluation=True, manual=False, teacher_base_url=None, teacher_api_key=None, teacher_model=None):
    """Generates a single trajectory for a given task."""
    print(f"--- Starting generation for Task ID: {task_id} ---")
    print(f"Task: {task_description}")
    print(f"Setup commands: {setup_commands}")
    print(f"Difficulty level: {difficulty_level}, Realism: {how_realistic}")
    print(f"Required tools: {required_tools}")
    print(f"Success condition: {success_condition}")

    # Use provided teacher configuration or fall back to defaults
    if teacher_base_url is None:
        teacher_base_url = "https://api.deepseek.com"
    if teacher_api_key is None:
        teacher_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if teacher_model is None:
        teacher_model = "deepseek-chat"

    teacher = ShellTeacher(base_url=teacher_base_url, api_key=teacher_api_key, model=teacher_model)
    sandbox = Sandbox(setup_commands=setup_commands)
    judge = Judge()
    
    trajectory = []
    current_turn = 1
    evaluation = None
    
    # Start the secure sandbox environment
    sandbox.start()

    try:
        while current_turn <= 15: # Safety break to prevent infinite loops
            if manual:
                action = input(f"Turn {current_turn}: Enter command (or 'exit 0' to finish): ")
                thought = "Manual user input."
            else:
                # Get the next thought and action from the teacher model
                thought, action = teacher.get_next_step(task_description, trajectory)

            if "exit 0" in action.strip():
                print("User or model indicated task is complete.")
                break

            # Execute the action in the sandbox
            stdout, stderr, exit_code = sandbox.execute_command(action)
            observation = stdout + stderr

            if manual:
                print(observation)

            # Record the full turn
            turn_data = {
                "turn": current_turn,
                "thought": thought,
                "action": action,
                "observation": observation,
                "exit_code": exit_code
            }
            trajectory.append(turn_data)
            
            print(f"Turn {current_turn}: Action='{action.strip()}', ExitCode={exit_code}")

            if exit_code != 0:
                print("Command failed. Trajectory might continue with debugging steps.")

            current_turn += 1

        # Run success condition check if provided
        success_condition_passed = None
        success_condition_output = None
        if success_condition:
            try:
                # Run the success condition
                stdout, stderr, exit_code = sandbox.execute_command(success_condition)
                success_condition_output = stdout + stderr
                success_condition_passed = (exit_code == 0)
                print(f"Success condition result: exit_code={exit_code}, passed={success_condition_passed}")
            except Exception as e:
                success_condition_output = f"Error running success condition: {e}"
                success_condition_passed = False
                print(f"Error running success condition: {e}")

        if run_evaluation:
            print(f"--- Evaluating trajectory for Task ID: {task_id} ---")
            evaluation = judge.evaluate_trajectory(task_description, setup_commands, trajectory)
            print(f"Evaluation complete. Rating: {evaluation.rating}/5")
        else:
            print(f"--- Skipping evaluation for Task ID: {task_id} ---")

    finally:
        # Always ensure the sandbox is stopped and cleaned up
        sandbox.stop()
        print("--- Sandbox stopped. ---")

    return {
        "dataset_id": f"she_syn_{task_id}",
        "source": "manual" if manual else "synthetic_teacher_model_v1",
        "setup_commands": setup_commands,
        "task": task_description,
        "how_realistic": how_realistic,
        "difficulty_level": difficulty_level,
        "required_tools": required_tools,
        "success_condition": success_condition,
        "trajectory": trajectory,
        "evaluation": {
          "rating": evaluation.rating if evaluation else None,
          "reasoning": evaluation.reasoning if evaluation else "Evaluation did not run.",
          "success_condition_passed": success_condition_passed,
          "success_condition_output": success_condition_output
        }
    }

# Thread-safe file writing
write_lock = threading.Lock()

def write_trajectory_safely(trajectory_data, filename="dataset.jsonl"):
    """Thread-safe writing of trajectory data to JSONL file."""
    with write_lock:
        with open(filename, "a") as f:
            f.write(json.dumps(trajectory_data) + "\n")
            f.flush()  # Ensure immediate write
    print(f"--- Saved trajectory for Task ID: {trajectory_data['dataset_id']} ---\n")

def generate_and_save_trajectory(task_item, output_file, run_evaluation, manual, teacher_base_url=None, teacher_api_key=None, teacher_model=None):
    """Wrapper function that generates and saves a trajectory."""
    task_id = task_item['id']
    task_description = task_item['task']
    setup_commands = task_item['setup_commands']
    how_realistic = task_item['how_realistic']
    difficulty_level = task_item['difficulty_level']
    required_tools = task_item['required_tools']
    success_condition = task_item['success_condition']
    try:
        trajectory_data = generate_trajectory(task_id, task_description, setup_commands, how_realistic, difficulty_level, required_tools, success_condition, run_evaluation, manual, teacher_base_url, teacher_api_key, teacher_model)
        write_trajectory_safely(trajectory_data, output_file)
        return f"Completed {task_id}"
    except Exception as e:
        print(f"Error processing {task_id}: {e}")
        return f"Failed {task_id}: {e}"

def run_concurrent_generation(task_file="tasks.jsonl", max_workers=3, output_file="dataset.jsonl", limit=20, run_evaluation=True, manual=False, teacher_base_url=None, teacher_api_key=None, teacher_model=None):
    """Run trajectory generation with controlled concurrency."""
    curator = TaskCurator(task_file=task_file)
    tasks = curator.get_tasks(limit=limit)
    
    print(f"Starting concurrent generation with max {max_workers} workers...")
    print(f"Processing {len(tasks)} tasks from {task_file}")
    print(f"Output will be saved to {output_file}")
    if not run_evaluation:
        print("LLM-judge evaluation is disabled.")
    if manual:
        print("Manual mode is enabled. You will be prompted for commands.")

    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(generate_and_save_trajectory, task_item, output_file, run_evaluation, manual, teacher_base_url, teacher_api_key, teacher_model): task_item['id'] 
            for task_item in tasks
        }
        
        # Process completed tasks as they finish with progress bar
        from concurrent.futures import as_completed
        with tqdm(total=len(tasks), desc="Processing tasks", unit="task") as pbar:
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                    print(f"✅ {result}")
                except Exception as e:
                    print(f"❌ Task {task_id} failed: {e}")
                pbar.update(1)
    
    end_time = time.time()
    print(f"\n🎉 All tasks completed in {end_time - start_time:.1f} seconds")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate shell agent training trajectories")
    parser.add_argument(
        "--task-file", 
        type=str, 
        default="eval_split.jsonl",
        help="Path to the JSONL file containing tasks (default: eval_tasks.jsonl)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of concurrent workers (default: 3)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="dataset.jsonl",
        help="Path to output JSONL file for trajectories (default: dataset.jsonl)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of tasks to process (default: 20)"
    )
    parser.add_argument(
        "--run-evaluation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the LLM-judge evaluation. Enabled by default."
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Enable manual mode to override LLM actions with user input."
    )
    
    args = parser.parse_args()
    print(os.getenv("HTTP_PROXY"))

    teacher_base_url = "http://rearden:8000/v1"
    teacher_api_key = "MEOW"
    teacher_model = "deathbyknowledge/Qwen3-8B-Shell-SFT"

    max_workers = args.max_workers
    if args.manual:
        print("Manual mode enabled. Overriding max_workers to 1.")
        max_workers = 1

    # Run the concurrent generation
    run_concurrent_generation(
        task_file=args.task_file,
        max_workers=max_workers,
        output_file=args.output_file,
        limit=args.limit,
        run_evaluation=args.run_evaluation,
        manual=args.manual,
        teacher_base_url=teacher_base_url,
        teacher_api_key=teacher_api_key,
        teacher_model=teacher_model
    )

if __name__ == "__main__":
    main()

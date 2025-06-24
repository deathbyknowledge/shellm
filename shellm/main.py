# main.py
import json
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import argparse
from task_curator import TaskCurator
from teacher import Teacher
from sandbox import Sandbox
from judge import Judge
from dotenv import load_dotenv

load_dotenv()

def generate_trajectory(task_id, task_description, setup_commands):
    """Generates a single trajectory for a given task."""
    print(f"--- Starting generation for Task ID: {task_id} ---")
    print(f"Task: {task_description}")
    print(f"Setup commands: {setup_commands}")

    teacher = Teacher(base_url=None, api_key=None, model="gpt-4.1-2025-04-14")
    sandbox = Sandbox(setup_commands=setup_commands)
    judge = Judge()
    
    trajectory = []
    current_turn = 1
    evaluation = None
    
    # Start the secure sandbox environment
    sandbox.start()

    try:
        while current_turn <= 15: # Safety break to prevent infinite loops
            # Get the next thought and action from the teacher model
            thought, action = teacher.get_next_step(task_description, trajectory)

            if action.strip() == "exit 0":
                print("Teacher model indicated task is complete.")
                break

            # Execute the action in the sandbox
            stdout, stderr, exit_code = sandbox.execute_command(action)
            observation = stdout + stderr

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

        print(f"--- Evaluating trajectory for Task ID: {task_id} ---")
        evaluation = judge.evaluate_trajectory(task_description, setup_commands, trajectory)
        print(f"Evaluation complete. Rating: {evaluation.rating}/5")

    finally:
        # Always ensure the sandbox is stopped and cleaned up
        sandbox.stop()
        print("--- Sandbox stopped. ---")

    return {
        "dataset_id": f"she_syn_{task_id}",
        "source": "synthetic_teacher_model_v1",
        "setup_commands": setup_commands,
        "task": task_description,
        "trajectory": trajectory,
        "evaluation": {
          "rating": evaluation.rating if evaluation else None,
          "reasoning": evaluation.reasoning if evaluation else "Evaluation did not run."
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

def generate_and_save_trajectory(task_item, output_file):
    """Wrapper function that generates and saves a trajectory."""
    task_id = task_item['id']
    task_description = task_item['description']
    setup_commands = task_item['setup_commands']
    try:
        trajectory_data = generate_trajectory(task_id, task_description, setup_commands)
        write_trajectory_safely(trajectory_data, output_file)
        return f"Completed {task_id}"
    except Exception as e:
        print(f"Error processing {task_id}: {e}")
        return f"Failed {task_id}: {e}"

def run_concurrent_generation(task_file="tasks.jsonl", max_workers=3, output_file="dataset.jsonl", limit=20):
    """Run trajectory generation with controlled concurrency."""
    curator = TaskCurator(task_file=task_file)
    tasks = curator.get_tasks(limit=limit)
    
    print(f"Starting concurrent generation with max {max_workers} workers...")
    print(f"Processing {len(tasks)} tasks from {task_file}")
    print(f"Output will be saved to {output_file}")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(generate_and_save_trajectory, task_item, output_file): task_item['id'] 
            for task_item in tasks
        }
        
        # Process completed tasks as they finish
        for future in future_to_task:
            task_id = future_to_task[future]
            try:
                result = future.result()
                print(f"✅ {result}")
            except Exception as e:
                print(f"❌ Task {task_id} failed: {e}")
    
    end_time = time.time()
    print(f"\n🎉 All tasks completed in {end_time - start_time:.1f} seconds")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate shell agent training trajectories")
    parser.add_argument(
        "--task-file", 
        type=str, 
        default="eval_tasks.jsonl",
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
    
    args = parser.parse_args()
    
    # Run the concurrent generation
    run_concurrent_generation(
        task_file=args.task_file,
        max_workers=args.max_workers,
        output_file=args.output_file,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
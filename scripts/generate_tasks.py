import random
import instructor
import chromadb
import os
import json
import asyncio
import numpy as np
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

load_dotenv()

THRESHOLD = 0.95
CHROMA_DB_DIR = ".chroma_db"  # Directory for persistent ChromaDB storage
# Create embedding function using OpenAI
db_client = chromadb.PersistentClient(CHROMA_DB_DIR)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)
ALLOWED_TOOLS = ["ls", "pwd", "cat", "head", "tail", "touch", "mkdir", "mv", "cp", "rm", "grep", "wc", "sort", "uniq", "cut", "find", "echo", "chmod", "du", "df", "tar", "gzip", "gunzip", "tee", "cd", "rmdir"]

system_prompt = f"""
You are a meticulous system designer creating a benchmark for an AI agent. The agent operates in a sandboxed, non-interactive bash shell on a Debian-based Linux system. Your function is to generate a diverse set of tasks that test the agent's ability to use shell commands to solve problems.
Your output MUST be a JSON list of task objects. Do not output any text outside of the JSON structure. Don't wrap your output in ``` either.

Each task must be a self-contained problem solvable in a non-interactive shell. The task descriptions should be natural language instructions a user would give, without explicitly naming the commands required for the solution. Do not create "dummy" names (e.g. "deletable_project"), instead build realistic scenarios.

Difficulty Level Definitions:
- Level 1 (Basic Operations): Requires a single, simple command.
- Level 2 (Simple Composition): Requires a simple pipe (|) or redirection (>) involving 2-3 commands.
- Level 3 (Multi-Step Logic): Requires a sequence of 2-4 distinct commands where the output of one informs the next.
- Level 4 (Complex Manipulation): Requires complex pipes or multiple steps to parse, transform, and extract specific information from text.

Environment Constraints:
- Allowed Tools: {", ".join(random.sample(ALLOWED_TOOLS, len(ALLOWED_TOOLS)))}
- Forbidden Commands: The environment is strictly non-interactive. Do not generate tasks that would require interactive tools such as vim, nano, less, more, top, man, ssh, ftp or any command that prompts for user input (e.g., read).

All generated tasks must adhere to the following JSON schema for each object in the list:
- description: A clear, natural language instruction for the AI agent.
- difficulty_level: An integer from 1 to 4, based on the definitions provided below.
- how_realistic: A float between 0.0 and 1.0, estimating the likelihood a real user would request this task.
- setup_commands: A list of bash commands that create the necessary files/directories for the task. Use realistic, non-obvious names for files and directories (e.g., use "apollo-ingest-service" instead of "my_project").
- required_tools: (For researcher analysis only). A list of the essential shell commands from the "Allowed Tools" list that represent an intended solution path. This field is for post-experiment analysis to diagnose agent weaknesses and is never shown to the agent.
- success_condition: (For automated evaluation). A single, robust bash command that exits with code 0 on success and a non-zero code on failure. This is essential for programmatically verifying task completion and for providing reward signals in reinforcement learning.

Example output:
{{
  "description": "The application 'kronos-scheduler' has written too many logs. Find all `.log` files within its directory that were modified more than 7 days ago and compress them into a single tarball named 'old_logs.tar.gz' in the `/tmp` directory.",
  "difficulty_level": 4,
  "how_realistic": 0.85,
  "setup_commands": [
    "mkdir -p /app/kronos-scheduler/logs",
    "touch -d '10 days ago' /app/kronos-scheduler/logs/events_2024-05-10.log",
    "touch -d '9 days ago' /app/kronos-scheduler/logs/errors_2024-05-11.log",
    "touch -d '2 days ago' /app/kronos-scheduler/logs/events_2024-05-18.log",
    "echo 'data' > /app/kronos-scheduler/logs/events_2024-05-10.log"
  ],
  "required_tools": ["find", "tar"],
  "success_condition": "tar -tzf /tmp/old_logs.tar.gz | grep -q 'events_2024-05-10.log' && tar -tzf /tmp/old_logs.tar.gz | grep -q 'errors_2024-05-11.log' && ! tar -tzf /tmp/old_logs.tar.gz | grep -q 'events_2024-05-18.log'"
}}
"""

class Task(BaseModel):
    description: str
    difficulty_level: int
    setup_commands: list[str]
    how_realistic: float
    required_tools: list[str]
    success_condition: str

class TaskBatch(BaseModel):
    tasks: list[Task]

def cosine_similarity(embeddings):
    """Simple cosine similarity implementation using numpy."""
    embeddings = np.array(embeddings)
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    # Compute cosine similarity matrix
    return np.dot(normalized_embeddings, normalized_embeddings.T)

class TaskGenerator:
    def __init__(self, model="deepseek-chat", api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"):
        # Keep sync client for embeddings and ChromaDB operations
        self.sync_client = instructor.from_openai(OpenAI(api_key=api_key, base_url=base_url))
        # Add async client for parallel task generation
        self.async_client = instructor.from_openai(AsyncOpenAI(api_key=api_key, base_url=base_url))
        self.model = model
        self.db_client = db_client
        self.collection = self.db_client.get_or_create_collection("task_descriptions", embedding_function=openai_ef, metadata={"hnsw:space": "cosine"}) # type: ignore
        # OpenAI client for embeddings (for batch deduplication)
        self.openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def generate_single_batch(self, batch_size=5, recent_descriptions=None):
        """Generates a single batch of tasks asynchronously."""
        prompt = f"Generate {batch_size} new tasks. You must phrase them differently, don't start all tasks the same (e.g. \"The file ...\")."
        if recent_descriptions:
            prompt += f"\n **CRITICAL**: To ensure diversity, please AVOID generating tasks that are semantically similar to the following examples we have already collected:\n{'\n'.join([f'- {desc}' for desc in recent_descriptions])}."
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_model=TaskBatch,
                temperature=0.7,
            )
            return response.tasks if response else []
        except Exception as e:
            print(f"Error generating batch: {e}")
            return []

    async def get_embeddings(self, texts):
        """Get embeddings for a list of texts."""
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None

    def find_batch_duplicates(self, tasks_with_embeddings, threshold=THRESHOLD):
        """Find and remove duplicates within a batch using embeddings."""
        if len(tasks_with_embeddings) <= 1:
            return [task for task, _ in tasks_with_embeddings]

        embeddings = [embedding for _, embedding in tasks_with_embeddings]
        similarity_matrix = cosine_similarity(embeddings)
        
        unique_tasks = []
        used_indices = set()
        
        for i, (task, _) in enumerate(tasks_with_embeddings):
            if i in used_indices:
                continue
                
            unique_tasks.append(task)
            used_indices.add(i)
            
            # Mark similar tasks as duplicates
            for j in range(i + 1, len(tasks_with_embeddings)):
                if j not in used_indices and similarity_matrix[i][j] > threshold:
                    print(f"Batch duplicate detected: '{tasks_with_embeddings[j][0].description}' -> '{task.description}'")
                    used_indices.add(j)
        
        return unique_tasks

    async def generate_and_filter_tasks_parallel(self, num_batches=3, batch_size=5):
        """Generates multiple batches in parallel and filters out duplicates."""
        print(f"Generating {num_batches} batches of {batch_size} tasks in parallel...")
        
        # Get recent tasks for context
        recent_descriptions = []
        collection_length = self.collection.count()
        if collection_length > 0:
            sample_size = min(batch_size, collection_length)
            random_ids = random.sample(range(collection_length), sample_size)
            recent_tasks = [self.collection.get(ids=[str(id)]) for id in random_ids]
            for result in recent_tasks:
                if result['documents']:
                    recent_descriptions.extend(result['documents'])

        # Generate batches in parallel
        batch_tasks = await asyncio.gather(*[
            self.generate_single_batch(batch_size, recent_descriptions)
            for _ in range(num_batches)
        ])
        
        # Flatten all tasks
        all_tasks = []
        for batch in batch_tasks:
            all_tasks.extend(batch)
        
        if not all_tasks:
            print("No tasks generated.")
            return []

        print(f"Generated {len(all_tasks)} total tasks across {num_batches} batches.")
        
        # Get embeddings for all tasks
        task_descriptions = [task.description for task in all_tasks]
        embeddings = await self.get_embeddings(task_descriptions)
        
        if not embeddings:
            print("Failed to get embeddings.")
            return []

        # Remove batch duplicates
        tasks_with_embeddings = list(zip(all_tasks, embeddings))
        unique_batch_tasks = self.find_batch_duplicates(tasks_with_embeddings, threshold=THRESHOLD)
        
        print(f"After batch deduplication: {len(unique_batch_tasks)} unique tasks.")
        
        # Check remaining tasks against ChromaDB
        final_unique_tasks = []
        for task in unique_batch_tasks:
            if self._is_unique(task.description):
                final_unique_tasks.append(task)
                self._add_to_collection(task.description)
        
        print(f"After ChromaDB deduplication: {len(final_unique_tasks)} final unique tasks.")
        return final_unique_tasks

    def _is_unique(self, description, threshold=THRESHOLD):
        """Checks if a task description is unique based on semantic similarity."""
        if self.collection.count() == 0:
            return True

        results = self.collection.query(
            query_texts=[description],
            n_results=1
        )
        
        if not results or not results.get('distances') or not results['distances'] or len(results['distances'][0]) == 0:
            return True

        # ChromaDB returns distances, not similarities. A smaller distance means more similar.
        # The distance is squared L2, so we need to be careful with the threshold.
        # For now, we'll assume a simple distance check.
        if results['distances'][0][0] < (1 - threshold):
            print(f"Duplicate task detected: '{description}'")
            return False
            
        return True

    def _add_to_collection(self, description):
        """Adds a task description to the ChromaDB collection."""
        # Use a simple counter for the ID
        new_id = str(self.collection.count() + 1)
        self.collection.add(
            ids=[new_id],
            documents=[description]
        )

def save_tasks_to_jsonl(tasks, filename="tasks.jsonl"):
    """Appends a list of tasks to a JSONL file."""
    with open(filename, "a") as f:
        for i, task in enumerate(tasks):
            task_data = task.model_dump()
            task_data['id'] = f"task_{int(time.time())}_{i}"
            f.write(json.dumps(task_data) + "\n")

if __name__ == "__main__":
    import time
    generator = TaskGenerator()
    
    try:
        while True:
            # Use the new parallel method
            tasks = asyncio.run(generator.generate_and_filter_tasks_parallel(num_batches=10, batch_size=5))
            if tasks:
                save_tasks_to_jsonl(tasks)
            print("Waiting for 2 seconds before the next batch...")
            time.sleep(2)  # Increased wait time since we're generating more tasks per iteration
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")

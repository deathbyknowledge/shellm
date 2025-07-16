import json
import sys

def convert_to_openai_format(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            task = data["task"]
            trajectory = data["trajectory"]
            id = data["dataset_id"]
            setup_commands = data["setup_commands"]
            success_condition = data["success_condition"]
            prompt = [{"role": "system", "content": task}]
            messages = []
            for turn in trajectory:
                thought = turn["thought"]
                action = turn["action"]
                observation = turn["observation"]
                messages.append({"role": "assistant", "content": "# " + thought})
                messages.append({"role": "user", "content": ""})
                messages.append({"role": "assistant", "content": action})
                messages.append({"role": "user", "content": observation})
            messages.append({"role": "assistant", "content": "exit 0"})
            output_data = {"prompt": prompt, "completion": messages, "id": id, "setup_commands": setup_commands, "success_condition": success_condition}
            outfile.write(json.dumps(output_data) + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input.jsonl output.jsonl")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_to_openai_format(input_file, output_file)
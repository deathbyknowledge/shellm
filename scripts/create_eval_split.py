#!/usr/bin/env python3

import json
import random
from collections import defaultdict
import argparse

def load_dataset(filename):
    """Load the dataset and group by difficulty level, filtering for rating == 5."""
    entries_by_level = defaultdict(list)
    total_entries = 0
    rating_5_entries = 0
    
    with open(filename, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                level = entry.get('difficulty_level')
                rating = entry.get('evaluation', {}).get('rating')
                
                if level:
                    total_entries += 1
                    
                    # Only include entries with perfect rating (5) and valid difficulty level
                    if rating == 5:
                        rating_5_entries += 1
                        entries_by_level[level].append(entry)
            except json.JSONDecodeError:
                continue
    
    print(f"Filtered dataset: {rating_5_entries}/{total_entries} entries have rating == 5")
    return entries_by_level

def create_eval_split(entries_by_level, samples_per_level):
    """Create eval split by sampling specified number of entries per level."""
    eval_entries = []
    
    for level, count in samples_per_level.items():
        available = entries_by_level.get(level, [])
        if len(available) < count:
            print(f"Warning: Only {len(available)} entries available for level {level}, requested {count}")
            count = len(available)
        
        # Randomly sample entries
        sampled = random.sample(available, count)
        eval_entries.extend(sampled)
        
        print(f"Level {level}: sampled {count} entries")
    
    return eval_entries

def save_eval_split(eval_entries, output_file):
    """Save the eval split to a JSONL file."""
    with open(output_file, 'w') as f:
        for entry in eval_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Saved {len(eval_entries)} entries to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Create evaluation split from dataset")
    parser.add_argument("--input", default="v3-simple-unix.jsonl", help="Input dataset file")
    parser.add_argument("--output", default="eval_split.jsonl", help="Output eval split file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Default sampling strategy
    parser.add_argument("--level1", type=int, default=5, help="Number of level 1 entries")
    parser.add_argument("--level2", type=int, default=4, help="Number of level 2 entries")
    parser.add_argument("--level3", type=int, default=3, help="Number of level 3 entries")
    parser.add_argument("--level4", type=int, default=2, help="Number of level 4 entries")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Load dataset
    print(f"Loading dataset from {args.input}...")
    entries_by_level = load_dataset(args.input)
    
    # Print dataset statistics
    print("\nDataset statistics:")
    for level in sorted(entries_by_level.keys()):
        print(f"  Level {level}: {len(entries_by_level[level])} entries")
    
    # Define sampling strategy
    samples_per_level = {
        1: args.level1,
        2: args.level2,
        3: args.level3,
        4: args.level4
    }
    
    print(f"\nSampling strategy:")
    for level, count in samples_per_level.items():
        print(f"  Level {level}: {count} entries")
    
    # Create eval split
    print(f"\nCreating eval split...")
    eval_entries = create_eval_split(entries_by_level, samples_per_level)
    
    # Save eval split
    save_eval_split(eval_entries, args.output)
    
    print(f"\nEval split created successfully!")
    print(f"Total entries in eval split: {len(eval_entries)}")

if __name__ == "__main__":
    main() 
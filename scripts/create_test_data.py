#!/usr/bin/env python3
"""
Script to create test data from the APPS dataset.
"""
import json
from pathlib import Path
import pandas as pd

# Read the APPS dataset
APPS_CSV_PATH = "/Users/julianstastny/Code/exploration-hacking/apps_dataset/apps_test_training.csv"
OUTPUT_DIR = Path("test_data")

def create_test_data() -> None:
    """Extract and save test data from the APPS dataset."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Read the CSV
    df = pd.read_csv(APPS_CSV_PATH)
    
    # Find problems with specific formats
    call_based = df[df['user_prompt'].str.contains('IMPORTANT: Use Call-Based format', na=False)]
    std_input = df[df['user_prompt'].str.contains('IMPORTANT: Use Standard Input format', na=False)]
    
    # Take 2 examples of each
    call_based_sample = call_based.head(2)
    std_input_sample = std_input.head(2)
    
    # Process all samples
    for idx, row in call_based_sample.iterrows():
        problem_dir = OUTPUT_DIR / f"problem_{row['problem_id']}"
        problem_dir.mkdir(exist_ok=True)
        
        # Save question
        with open(problem_dir / "question.txt", 'w') as f:
            f.write(row['question'])
        
        # Save solutions
        solutions = json.loads(row['solutions'])
        with open(problem_dir / "solutions.json", 'w') as f:
            json.dump(solutions, f, indent=2)
        
        # Save input/output
        input_output = json.loads(row['input_output'])
        with open(problem_dir / "input_output.json", 'w') as f:
            json.dump(input_output, f, indent=2)
        
        # Save starter code if present
        if pd.notna(row.get('starter_code')):
            with open(problem_dir / "starter.py", 'w') as f:
                f.write(row['starter_code'])
    
    for idx, row in std_input_sample.iterrows():
        problem_dir = OUTPUT_DIR / f"problem_{row['problem_id']}"
        problem_dir.mkdir(exist_ok=True)
        
        # Save question
        with open(problem_dir / "question.txt", 'w') as f:
            f.write(row['question'])
        
        # Save solutions
        solutions = json.loads(row['solutions'])
        with open(problem_dir / "solutions.json", 'w') as f:
            json.dump(solutions, f, indent=2)
        
        # Save input/output
        input_output = json.loads(row['input_output'])
        with open(problem_dir / "input_output.json", 'w') as f:
            json.dump(input_output, f, indent=2)
        
        # Save starter code if present
        if pd.notna(row.get('starter_code')):
            with open(problem_dir / "starter.py", 'w') as f:
                f.write(row['starter_code'])

if __name__ == "__main__":
    create_test_data() 
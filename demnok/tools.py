import json
def combine_jsonl_files(file1_path, file2_path, output_path):
    """
    Combine two JSONL files based on matching qid and question_id
    
    Args:
        file1_path: Path to file with {"qid": xxx, "question": xxxx, "answers": xxxx}
        file2_path: Path to file with {"question_id": xxx, "topk_ids": xxx}
        output_path: Path for the combined output file
    """
    
    # Read first file and create a dictionary with qid as key
    qid_data = {}
    with open(file1_path, 'r', encoding='utf-8') as f1:
        for line in f1:
            data = json.loads(line.strip())
            qid_data[data['qid']] = data
    
    # Read second file and combine with first file data
    combined_data = []
    with open(file2_path, 'r', encoding='utf-8') as f2:
        for line in f2:
            data = json.loads(line.strip())
            question_id = data['question_id']
            
            # If we find a matching qid in the first file
            if question_id in qid_data:
                # Combine the data from both files
                combined_record = {**qid_data[question_id], **data}
                # Remove duplicate id field (keeping qid, removing question_id)
                del combined_record['question_id']
                
                combined_data.append(combined_record)
            else:
                print(f"Warning: No matching qid found for question_id: {question_id}")
    
    # Write combined data to output file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for record in combined_data:
            output_file.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Combined {len(combined_data)} records and saved to {output_path}")

import json
from typing import List, Dict, Any

def read_jsonl_file(file_path: str) -> List[Dict[Any, Any]]:
    """
    Read a JSONL file and return a list of dictionaries.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries from the JSONL file
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse JSON on line {line_number}: {e}")
                        print(f"Problematic line: {line}")
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return []

def reorder_jsonl_data(data: List[Dict[Any, Any]], reordered_indices: List[int]) -> List[Dict[Any, Any]]:
    """
    Reorder the list of dictionaries based on the given indices.
    
    Args:
        data: List of dictionaries from JSONL file
        reordered_indices: List of indices indicating the new order
        
    Returns:
        Reordered list of dictionaries
        
    Example:
        If original data is [item0, item1, item2] and reordered_indices is [2, 0, 1],
        the result will be [item2, item0, item1]
    """
    if not data:
        print("Warning: Data list is empty.")
        return []
    
    if not reordered_indices:
        print("Warning: Reordered indices list is empty.")
        return data
    
    # Validate indices
    max_index = len(data) - 1
    invalid_indices = [idx for idx in reordered_indices if idx < 0 or idx > max_index]
    
    if invalid_indices:
        print(f"Error: Invalid indices found: {invalid_indices}. Data has {len(data)} items (indices 0-{max_index}).")
        return data
    
    if len(reordered_indices) != len(data):
        print(f"Warning: Length mismatch. Data has {len(data)} items but {len(reordered_indices)} indices provided.")
        print("Using available indices only.")
    
    # Reorder the data
    try:
        reordered_data = []
        used_indices = set()
        
        for idx in reordered_indices:
            if idx in used_indices:
                print(f"Warning: Duplicate index {idx} found. Skipping duplicate.")
                continue
            if 0 <= idx < len(data):
                reordered_data.append(data[idx])
                used_indices.add(idx)
            else:
                print(f"Warning: Index {idx} is out of range. Skipping.")
        
        return reordered_data
        
    except Exception as e:
        print(f"Error during reordering: {e}")
        return data

def write_jsonl_file(data: List[Dict[Any, Any]], output_path: str) -> bool:
    """
    Write a list of dictionaries to a JSONL file.
    
    Args:
        data: List of dictionaries to write
        output_path: Path for the output JSONL file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)
                file.write(json_line + '\n')
        return True
    except Exception as e:
        print(f"Error writing to file '{output_path}': {e}")
        return False

def read_and_reorder_jsonl(input_file_path: str, reordered_indices: List[int], output_file_path: str = None) -> List[Dict[Any, Any]]:
    """
    Complete function to read JSONL, reorder based on indices, and optionally save to new file.
    
    Args:
        input_file_path: Path to the input JSONL file
        reordered_indices: List of indices for reordering
        output_file_path: Optional path to save reordered data
        
    Returns:
        Reordered list of dictionaries
    """
    print(f"Reading JSONL file: {input_file_path}")
    
    # Read the JSONL file
    data = read_jsonl_file(input_file_path)
    
    if not data:
        print("No data to reorder.")
        return []
    
    print(f"Successfully read {len(data)} items from JSONL file.")
    
    # Reorder the data
    print(f"Reordering data based on {len(reordered_indices)} indices...")
    reordered_data = reorder_jsonl_data(data, reordered_indices)
    
    print(f"Reordering complete. Result has {len(reordered_data)} items.")
    
    # Optionally save to output file
    if output_file_path:
        print(f"Saving reordered data to: {output_file_path}")
        if write_jsonl_file(reordered_data, output_file_path):
            print("Successfully saved reordered data.")
        else:
            print("Failed to save reordered data.")
    
    return reordered_data
from typing import List, Tuple, Union, Optional
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def get_harmful_instructions(hf_path: Optional[str] = 'Undi95/orthogonal-activation-steering-TOXIC', 
                             additional_instructions: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    """
    Loads a dataset containing harmful instructions and splits it into training and testing sets.
    
    The dataset is loaded from the Hugging Face Hub and consists of harmful prompts.
    
    Args:
        hf_path (Optional[str]): The Hugging Face dataset path. Defaults to 'Undi95/orthogonal-activation-steering-TOXIC'.
        additional_instructions (Optional[List[str]]): Additional harmful instructions to be appended to the training set.
    
    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists - the training and testing sets of harmful instructions.
    """
    dataset = load_dataset(hf_path)
    
    # Extract the 'goal' field which contains the harmful instructions
    instructions = [i['goal'] for i in dataset['test']]
    
    # Split the dataset into training (80%) and testing (20%)
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    
    # Append additional instructions if provided
    if additional_instructions:
        train.extend(additional_instructions)
    
    return train, test

def get_harmless_instructions(hf_path: Optional[str] = 'tatsu-lab/alpaca', 
                               additional_instructions: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    """
    Loads a dataset containing harmless instructions and filters out those with additional inputs.
    
    The dataset is sourced from the Hugging Face Hub and consists of safe instructions. Only instructions 
    that do not have accompanying inputs are retained to ensure purity of intent.
    
    Args:
        hf_path (Optional[str]): The Hugging Face dataset path. Defaults to 'tatsu-lab/alpaca'.
        additional_instructions (Optional[List[str]]): Additional harmless instructions to be appended to the training set.
    
    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists - the training and testing sets of harmless instructions.
    """
    dataset = load_dataset(hf_path)
    
    # Extract instructions that do not have additional inputs
    instructions = [
        i['instruction'] for i in dataset['train'] if i['input'].strip() == ''
    ]
    
    # Split the dataset into training (80%) and testing (20%)
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    
    # Append additional instructions if provided
    if additional_instructions:
        train.extend(additional_instructions)
    
    return train, test

def prepare_dataset(dataset: Union[Tuple[List[str], List[str]], List[str]]) -> Tuple[List[str], List[str]]:
    """
    Ensures a dataset is properly split into training and testing sets.
    
    If the dataset is not already split, it is divided into a 90/10 split.
    
    Args:
        dataset (Union[Tuple[List[str], List[str]], List[str]]):
            A dataset that is either already split into (train, test) or a single list requiring a split.
    
    Returns:
        Tuple[List[str], List[str]]: A tuple containing the training and testing sets.
    """
    if len(dataset) != 2:
        # If dataset is a single list, split into 90% training and 10% testing
        train, test = train_test_split(dataset, test_size=0.1, random_state=42)
    else:
        # If dataset is already split, use it as-is
        train, test = dataset
    return train, test

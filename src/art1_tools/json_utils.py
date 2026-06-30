import os
import sys
import numpy as np
from typing import List, Tuple

class JsonReader:
    """A class to read and process JSON files.

    This class provides methods to read a JSON file line by line, extract float values based on a specified tree structure, 
    and parse these values into nested lists.
        json_path (str): Path to the JSON file.
        dim_count (int): Counter for the dimensions in the JSON structure.
        record_flag (bool): Flag to indicate if the current chunk should be recorded.
        tree_keys (List[str]): List of keys representing the tree structure.
        chunks_fill (bool): Flag to indicate if chunks have been filled.
        n_line (int): Current line number being processed.
    Methods:
        process_json_file(tree_keys: str | List[str]) -> List[float]:
        get_chunks_from_tree(chunk: str) -> Tuple[List[str], bool, List[str]]:
            Get the chunks of data from the tree.
        switch_record_flag(chunk: str) -> bool:
            Switch the record flag based on the presence of tree keys in the chunk.
        save_chunks(chunk: str, record_flag: bool, dim_count: int) -> List[str]:
            Save chunks of data based on a condition.
        chunks_to_floats(saved_chunks: List[str]) -> List[float]:
            Convert saved chunks of data to float values.
        parse_nested_list(original_list: List) -> List[List[float]]:
            Parse a nested list from a flat list with delimiters '[' and ']'.
    """

    def __init__(self, json_path: str):
        """Initialize the JsonReader class.
        """
        self.json_path = json_path
        self.dim_count = 0
        self.record_flag = False

    def process_json_file(
        self,
        tree_keys: str | List[str],
        ) -> List[float]:
        """
        Process a JSON file line by line and extract float values based on the tree structure.

        Args:
            json_path: Path to the JSON file.
            tree_keys: List of keys representing the tree structure.

        Returns:
            A list of float values extracted from the JSON file.
        """
        saved_chunks = []
        past_chunk = ''
        dim_count = 1
        if isinstance(tree_keys, str):
            self.tree_keys = tree_keys.split('.')

        with open(self.json_path) as file:
            for n, line in enumerate(file):
                chunk = self.get_chunks_from_tree(
                    chunk=line, 
                    )
                past_chunk = chunk
                saved_chunks += chunk
                self.chunks_fill = bool(saved_chunks)
                self.n_line = n

                #if print_shape_flag:
                    #print(len(saved_chunks))
                if not self.record_flag and self.chunks_fill:
                    break

        return  self.chunks_to_floats(saved_chunks)

    def get_chunks_from_tree(
        self,
        chunk: str, 
        ) -> Tuple[List[str], bool, List[str]]:
        """Get the chunks of data from the tree.
        
        This function retrieves the chunks of data from the tree and returns them.
        
        Args:
            chunk: A chunk of data to be processed.
            tree_keys: A list of keys representing the tree structure.
        
        Returns:
            A tuple with the remaining tree keys, a flag indicating if saving should stop, and the saved chunks.
        """
        dim_count = self.dim_count
        saved_chunk = []

        if not self.record_flag:
            self.record_flag = self.switch_record_flag(chunk)
                
        if not self.record_flag:
            chunk = []
        else:
            chunk, record_flag = self.save_chunks(chunk, self.record_flag, dim_count)
            self.record_flag = record_flag
        
        return chunk

    def switch_record_flag(
        self,
        chunk: str,
        ) -> bool:
        if len(self.tree_keys) == 1:
            key = self.tree_keys[0]
            if key not in chunk:
                record_flag = False
            if key in chunk:  
                print(f'finded: {key} in {chunk} at line: {self.n_line}')  
                record_flag = True
                print("recording...")

        else:
            for key in self.tree_keys:
                record_flag = False
                if key in chunk:  
                    print(f'finded: {key} in {chunk} at line: {self.n_line}')              
                    self.tree_keys.remove(key)
                else:
                    break
        return record_flag
    
    def save_chunks(
        self,
        chunk: str, 
        record_flag: bool, 
        dim_count: int
        ) -> List[str]:
        """Save chunks of data based on a condition.
        
        This function retrieves the chunks of data that are between two strings. 
        The start string triggers the saving of chunks, and the end string stops it.

        Args:
            chunk: A chunk of data to be processed.

        Returns:
            A tuple with the stop_saving flag and the saved chunks.
        """
        if record_flag:
            if '[' in chunk:
                self.dim_count += 1
            if ']' in chunk:
                self.dim_count -= 1
            saved_chunk = [chunk]
            if self.dim_count == 0:
                record_flag = False
        else:
            saved_chunk = []
        return saved_chunk, record_flag

    def chunks_to_floats(self, saved_chunks: List[str]) -> List[float]:
            float_chunks = []
            for line in saved_chunks:
                if '[' in line:
                    float_chunks.append('[')
                elif ']' in line:
                    float_chunks.append(']')
                else:
                    try:
                        float_chunks.append(float(line.strip().strip(',')))
                    except ValueError as e:
                        print(f'Error: {e}')
                        pass

            return self.parse_nested_list(float_chunks)

    def parse_nested_list(self, original_list: List) -> List[List[float]]:
        """
        Parses a nested list from a flat list with delimiters '[' and ']'.

        Args:
            original_list: The flat list containing nested list elements.

        Returns:
            A list of lists containing the parsed nested lists.
        """
        list_of_lists = []
        current_list = []

        for item in original_list:
            if item == '[':
                current_list = []
            elif item == ']':
                list_of_lists.append(current_list)
            else:
                current_list.append(item)

        return list_of_lists

def save_results(arr: np.ndarray, file_name: str) -> str:
    """
    Save the arr array to a file if it doesn't already exist.

    Args:
        arr (np.ndarray): The array to save.

    Returns:
        str: The path to the saved file.
    """

    # Obtener el nombre del usuario actual
    if sys.platform.startswith("linux"):
        usuario_actual = os.path.expanduser("~").split('/')[-1]
    elif sys.platform.startswith("win"):
        usuario_actual = os.path.expanduser("~").split('\\')[-1]
    else:
        raise EnvironmentError("Unsupported OS")
    # save the array
    arr_path = f'/home/{usuario_actual}/Doctorado/results/{file_name}.npy'
    if not os.path.exists(arr_path):
        np.save(arr_path, arr)
        
        print(f"The array was saved in {arr_path}")
    
    else:
        raise FileExistsError(f"The file {arr_path} already exists.")
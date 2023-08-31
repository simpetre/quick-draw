import os
import numpy as np
from torch.utils.data import Dataset
import json

class QuickDrawDataset(Dataset):
    """Dataset class for the QuickDraw data.
    
    Attributes:
        img_dir (str): Directory containing .npy files.
        filenames (list): List of filenames in img_dir.
        idx_to_file_map (list): Mapping from an index to its corresponding file and position within the file.
        total_samples (int): Total number of samples in the dataset.
    """
    
    def __init__(self, img_dir: str):
        """
        Initialize the QuickDrawDataset.
        
        Args:
            img_dir (str): Directory containing .npy files.
        """
        self.img_dir = img_dir
        self.filenames = [filename for filename in os.listdir(img_dir) if filename.endswith(".npy")]
        
        if not self.filenames:
            raise ValueError(f"No .npy files found in {img_dir}.")

        self._prepare_idx_to_file_map()
        self.total_samples = len(self.idx_to_file_map)

    def _prepare_idx_to_file_map(self):
        """Prepare the index to file mapping."""
        self.idx_to_file_map = []
        for label, filename in enumerate(self.filenames):
            num_samples = np.load(os.path.join(self.img_dir, filename)).shape[0]
            self.idx_to_file_map.extend([(label, i) for i in range(num_samples)])

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        """
        Fetch a sample given an index.
        
        Args:
            idx (int): Index of the sample to fetch.
            
        Returns:
            tuple: A sample and its corresponding label.
        """
        if idx >= self.total_samples or idx < -self.total_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.total_samples}.")
        
        label, file_idx = self.idx_to_file_map[idx]
        data_chunk = np.load(os.path.join(self.img_dir, self.filenames[label]), mmap_mode="r")
        sample = data_chunk[file_idx].astype('float32').reshape(-1, 28, 28)
        return sample, label

    def generate_class_index_to_name_json(self, json_filepath: str):
        """Generate a JSON file that maps class indices to class names.

        Args:
            json_filepath (str): Filepath to store the generated JSON file.
        """
        class_index_to_name = {i: os.path.splitext(name)[0] for i, name in enumerate(self.filenames)}
        with open(json_filepath, 'w') as json_file:
            json.dump(class_index_to_name, json_file)
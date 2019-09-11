from torch.utils.data import Dataset
import utils_char_dataset as utils
import numpy as np

class CharacterLanguageModelDataset(Dataset):
    """CharacterLanguageModelDataset Pickle Dataset"""

    def __init__(self, pickle_file, transform=None):
    
        self.dataset = utils.load_pickle(pickle_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx): 
        X_class_id_pad, Y_class_id_pad, pad_y_mask, len_x, len_y = self.dataset[idx]
        sample = {'X': X_class_id_pad, 'Y': Y_class_id_pad, 'label_mask': pad_y_mask, 
                  'len_x': len_x, 'len_y': len_y}

        # Handle Augmentations
        if self.transform:
            sample = self.transform(sample)

        return sample
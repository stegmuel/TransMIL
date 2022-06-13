from torchvision.transforms import ToTensor, Resize, Compose, RandomHorizontalFlip, RandomVerticalFlip, Normalize
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import torch

CLASS_DICT = {'0_N': 0, '1_PB': 1, '2_UDH': 2, '3_ADH': 3, '4_FEA': 4, '5_DCIS': 5, '6_IC': 6}


class BracsTilesDataset(Dataset):
    def __init__(self, queries):
        """
        Initializes the dataset class which relies on the assumption that the data is stored in a single root
        directory organized in sub-directories (one per class).
        phase and the second is either 'start' or 'stop'.
        """
        super(BracsTilesDataset, self).__init__()

        # Store basic parameters
        self.class_dict = CLASS_DICT
        self.filepaths = [f for q in queries for f in glob(q)]
        self.image_tiles = {}
        self.image_names = []
        for f in self.filepaths:
            image_name = '_'.join(f.split('/')[-1].split('.')[0].split('_')[:-1])
            if image_name in self.image_tiles.keys():
                self.image_tiles[image_name].append(f)
            else:
                self.image_tiles[image_name] = [f]
                self.image_names.append(image_name)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        # Get the image name
        image_name = self.image_names[index]
        
        # Load all the tiles of the image
        tiles = np.array([np.load(f) for f in self.image_tiles[image_name]])

        # Get the label
        label = CLASS_DICT[self.image_tiles[image_name][0].split('/')[-2]]
        return torch.tensor(tiles), torch.tensor(label)
        

if __name__ == '__main__':
    dataset = BracsTilesDataset(queries=['/media/thomas/Samsung_T5/BRACS/BRACS_bags/val/*/*.npy'])
    print(dataset.__getitem__(0)[1])
    
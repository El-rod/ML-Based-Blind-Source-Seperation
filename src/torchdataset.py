import os
import glob
import torch
import numpy as np

from torch.utils.data import Dataset


class RFMixtureDatasetBase(Dataset):
    """
    Converts all saved numpy RF-mixtures dataset to a pytorch Dataset
    (which was created by example_preprocess_npy_dataset.py)
    """
    def __init__(self, root_dir: str):
        """
        root_dir: root directory of the saved (numpy-type) dataset
        """
        super().__init__()
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError("Dataset root directory does not exsist.")
        self.files = glob.glob(os.path.join(self.root_dir, "*.npy"))

    def __len__(self):
        """
        returns number of files in the dataset (number of mixtures)
        """
        return len(self.files)

    def __getitem__(self, i):
        """
        returns the enumerated specified file as
        a dictionary with pytorch tensors:
        sample_mix, sample_soi
        """
        # np.load: load arrays or pickled objects from .npy, .npz or pickled files.
        data = np.load(self.files[i], allow_pickle=True).item()
        return {
            "sample_mix": torch.tensor(data["sample_mix"]).transpose(0, 1),
            "sample_soi": torch.tensor(data["sample_soi"]).transpose(0, 1),
        }


def get_train_val_dataset(dataset: Dataset, train_fraction: float):
    """
    dataset: pytorch dataset of signal
    train_fraction: what parentage to split the data (training and validation)

    returns training and validation dataset for the inserted dataset
    """
    # print(len(dataset)) <- was here priori
    val_examples = int((1 - train_fraction) * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - val_examples, val_examples], generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset


if __name__ == "__main__":
    # the reason this file is called example and not just what it is
    dataset = RFMixtureDatasetBase(
        root_dir="./npydataset/Dataset_QPSK_SynOFDM_Mixture",
    )

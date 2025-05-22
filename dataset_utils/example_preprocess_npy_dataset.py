"""
Used in conjunction with the Torch WaveNet training scripts;
the HDF5 files are processed into separate npy files (one file per mixture).
An associated dataloader is supplied within the PyTorch baseline code.
"""

import os, sys
import glob
import h5py
import numpy as np
from tqdm import tqdm

main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#print(main_folder)
#main_folder = r'/home/dsi/arielro1/tmp/rfproj/'

def preprocess_dataset(root_dir: str, save_dir: str) -> None:
    """
    root_dir: root parent directory to save the dataset
    save_dir: save folder directory name

    saves mixture as numpy file,
    in the format of a dictionary with pytorch tensors:
    sample_mix, sample_soi
    """
    save_dir = os.path.join(save_dir, os.path.basename(root_dir))
    os.makedirs(save_dir, exist_ok=True)
    # for f in tqdm(glob.glob(os.path.join(save_dir, "*.npy"))):
    #     os.remove(f)

    count = 0
    for folder in tqdm(glob.glob(os.path.join(root_dir, "*.h5"))):
        with h5py.File(folder, "r") as f:
            mixture = np.array(f.get("mixture"))
            soi = np.array(f.get("target"))
        for i in range(mixture.shape[0]):
            data = {
                "sample_mix": mixture[i, ...],
                "sample_soi": soi[i, ...],
            }
            np.save(os.path.join(save_dir, f"sample_{count}.npy"), data)
            count += 1


if __name__ == "__main__":
    # dataset_type = sys.argv[1]
    dataset_type = 'QPSK_Comm2andEMI1'
    # save_dir=f'{main_folder}/npydataset/Dataset_QPSK_Comm2andEMI1_Mixture/'
    # print(save_dir)
    # for f in tqdm(glob.glob(os.path.join(save_dir, "*.npy"))):
    #     os.remove(f)
    preprocess_dataset(root_dir=f'{main_folder}/dataset/Dataset_{dataset_type}_Mixture_rare30',
                       save_dir=f'{main_folder}/npydataset/')

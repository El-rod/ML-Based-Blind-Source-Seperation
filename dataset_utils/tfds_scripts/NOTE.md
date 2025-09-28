To build a tfds dataset out of the HDF5 files in your `/dataset` directory, run the following command in the terminal:

`tfds build dataset_utils/tfds_scripts/Dataset_SOI+IPM_Mixtures.py --data_dir serverdata/tfds/`

- where you replace `Dataset_SOI+IPM_Mixtures.py` to the correct script name that you desire.
- `/serverdata/tfds` is where the tfds dataset will be saved,
setup your Docker container so that `/serverdata` (or whatever folder name you choose) is link to your user's local 
directory on the server such i.e., `/localdata/my_user_dataset`.

You may notice some scripts have an explicit tfds supervised dataset by the line `supervised_keys=('mixture', 'signal')`, while other don't. This was done to generalize the tfds dataset, so the same tfds dataset could be used on both the UNet and CNN Detector without creating two different ones.

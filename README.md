# Machine-Learning-Based Blind Source Separation
- 4-th year's Final Project, for the partial fulfillment of the requirements for a B.Sc. Degree.

- Bar-Ilan University, the Faculty of Engineering, Project No. 507 (of year 2024).

## About this Repository

This code is a modification of the RF Challenge [starter code](https://github.com/RFChallenge/icassp2024rfchallenge), where it was updated with quality of life changes, and modified for simulations of training and testing signal mixture source separation via DNNs with interference uncertainty. The interference mixture is referred as the interference probability mixture (IPM), and the code supports K=2 types of interference.

## RF Challenge Dataset
For generating datasets of signal mixtures, the project used the [RF Challenge Interference Dataset](https://www.dropbox.com/scl/fi/zlvgxlhp8het8j8swchgg/dataset.zip?rlkey=4rrm2eyvjgi155ceg8gxb5fc4&dl=0).

## Weights
- QPSK CT-DNN weights for the UNet and WaveNet can be obtained here: [reference_models.zip](https://www.dropbox.com/scl/fi/890vztq67krephwyr0whb/reference_models.zip?rlkey=6yct3w8rx183f0l3ok2my6rej&dl=0)
(`/models` folder for UNet models, `/torchmodels` for WaveNet models).

- 8/16PSK CT-UNet weights, along with all MT-UNet and 8L-MT-UNet weights can be obtained [here](https://www.dropbox.com/scl/fi/gzriho4wv8zeodrswcejm/unet_models.zip?rlkey=qhejjcme6m88roktmgvlfkdyf&st=qhv4vs4t&dl=0).

- Extra weights of models that were also trained in the project can be obtained [here](https://www.dropbox.com/scl/fi/ijg3v1xfgxy7eu3vodf59/extra_models.zip?rlkey=glpxauktf53mpzy6bqr2smeck&st=n0nl2qne&dl=0).

# File Descriptions:

- For a complete overview of the dependencies within of the UNet's Anaconda environment, please refer [here (tf_env)](https://github.com/El-rod/ML-Based-Blind-Source-Seperation/blob/main/tf_env.yml).
- For a complete overview of the dependencies within of the WaveNet's Anaconda environment, please refer [here (pytorch_env)](https://github.com/El-rod/ML-Based-Blind-Source-Seperation/blob/main/pytorch_env.yml).

## Files for training:

(1) `/dataset_utils/generate_training_dataset.py`: python script that creates D sample mixtures with varying random target SINR levels (ranging between -33 dB and 3 dB). For each signal mixture configuration, the output is saved as D/n HDF5 files, each containing n mixtures. The project used the default RF Challenge setup of D=240000 and n=4000, making 60 HDF5 files for each mixture dataset.

####Note: `n_per_batch` is a misleading variable name, a more suitable one is `n_per_sinr`, but kept for legacy reasons.

(2) `/dataset_utils/tfds_scripts/`: each file in this folder preprocesses the training dataset HDF5 files created in (1) into a supervised-learning TensorFlow dataset for the UNet model. See the [note](https://github.com/El-rod/ML-Based-Blind-Source-Seperation/blob/main/dataset_utils/tfds_scripts/NOTE.md) in the folder for more infromation.

(3) `train_unet_model.py`: trains the Tensorflow UNet architecture (see `/src/unet_model.py` and `/src/unet_8layered_model.py`) on the tfds dataset created in a file from (2).

(4)  `/dataset_utils/preprocess_wavenet_training_dataset.py`: Used in conjunction with the Torch WaveNet supervised-learning training scripts; the HDF5 files are processed into separate npy files (one file per mixture). An [associated dataloader](https://github.com/El-rod/ML-Based-Blind-Source-Seperation/blob/main/src/torchdataset.py) is supplied within the PyTorch baseline code.

(5) `train_torchwavenet.py`: trains the PyTorch WaveNet architecture (see `/src/torchwavenet.py`), accompanied with dependencies including `supervised_config.yml` and the `/src/configs/` folder,  `src/torchdataset.py` as mentioned in (4), `src/learner_torchwavenet.py`, and `src/config_torchwavenet.py`.

(6) `train_cnn_detector.py`: trains the proof of concept CNN Detector model (see `/src/cnn_detector.py`) on the tfds dataset created in a file from (2).

## Files for testing
(1) `/dataset_utils/generate_mixture_testset_IPM.py`: generates a testset of a signal mixture with an interference mixture of P(b1)=p, of 11 discrete target SINR levels. Saves a pickle file `Dataset_Seed[seed]_[soi_type]+[interference_type1]∨[interference_type2].pkl` that contains `all_sig_mixture, all_sig1_groundtruth, all_bits1_groundtruth, meta_data`, sorted by SINR levels.

(2) `/dataset_utils/generate_mixture_testset_single_interference.py`: same as (1) but for p=1 only, i.e., saves it as `Dataset_Seed[seed_number]_[soi_type]+[interference_sig_type].pkl`.

(3) `evaltest_unet_IPM.py`: processes the testset created in (1) on the desired UNet model (CT-UNet, MT-UNet, 8L-MT-UNet), both waveform prediction and BER calculation. Saves results in the `/outputs` folder.

(4) `evaltest_unet_1interference.py`: same as (3) but for the testset created in (2). Here interfrence type 2 (p=0) is decided by which MT-UNet/8L-MT-UNet is chosen.

(5) `evaltest_wavenet_IPM.py`: processes the testset created in (1) on the desired WaveNet model (CT-WaveNet, MT-WaveNet), both waveform prediction and BER calculation. Saves results in the `/outputs` folder.

(6) `evaltest_wavenet_1interference.py`: same as (5) but for the testset created in (2). Here interfrence type 2 (p=0) is decided by which MT-WaveNet is chosen.

(7) `evaltest_cnn_detector.py`: contains the function for classifying the testset created in (1) by the CNN Detector into the predicted interference types. Has additional performance testing code similar to `plot_figure_save_results.py`.

## Utility files
(1). The SOI is generated by the files in the `/rfutils` folder.
   
(2). `plot_figure_save_results.py`: calculates the MSE and BER and saves them in a `.npz` file along with pyplots in the `/outputs` folder. Make sure the n_per_batch matches the one you generated in the testset!

# Project (Code-wise) Contributions

- Quality of Life Update of the RF Challenge starter code (added a thorough documentation, comments in every file, fixed some coding errors, updated the original RF Challenge Anaconda environments to have updated Python libraries versions – including code adjustments).
- Realized code of the proposed CT-DNN model (both oracle-detector and CNN-detector) and MT-DNN model, along with the 8L-MT-UNet variant.
- Generalized the RF Challenge starter code to both train and evaluate a signal mixture containing two possible interference signal types.
- Added support for 8PSK and 16PSK (Gray coded) SOI.




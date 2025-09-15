FROM continuumio/miniconda3
WORKDIR /app

RUN groupadd -g 1234 myusername && \
    useradd -u 9999 -g 1234 -m myusername
USER myusername

# COPY tf_env.yml .
# RUN conda env create --name tf_env --file=tf_env.yml
# RUN /bin/bash -c "source activate tf_env"
# RUN echo "source activate tf_env && export LD_LIBRARY_PATH=/home/myusername/.conda/envs/tf_env/lib/:$LD_LIBRARY_PATH" >> ~/.bashrc

COPY pytorch_env.yml .
RUN conda env create --name pytorch_env --file=pytorch_env.yml
RUN /bin/bash -c "conda init"
RUN /bin/bash -c "export LD_LIBRARY_PATH=/home/myusername/.conda/envs/pytorch_env/lib/:$LD_LIBRARY_PATH" >> ~/.bashrc
RUN /bin/bash -c "source activate pytorch_env" >> ~/.bashrc


# RUN conda create -n rf_tf python=3.9.20 -y && \
#     /bin/bash -c "source activate rf_tf && \
#     conda install -y numpy tqdm h5py && \
#     conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 && \
#     conda install -y -c conda-forge tensorflow-gpu && \
#     conda install -y -c nvidia cuda-nvcc && \
#     pip install sionna==0.10.0"
# RUN echo "source activate rf_tf && export LD_LIBRARY_PATH=/opt/conda/envs/rf_tf/lib:$LD_LIBRARY_PATH" >> ~/.bashrc

# RUN conda create -n rftorch python=3.11.11 -y && \
#    /bin/bash -c "source activate rftorch && \
#    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia && \
#    conda install -y numpy tqdm h5py && \
#    conda install -y -c conda-forge tensorflow-cpu=2.15.0 && \
#    conda install -y -c conda-forge omegaconf && \
#    pip install sionna==0.12.1"
# RUN echo "source activate rftorch" >> ~/.bashrc

# docker build -t ar_unet .
# docker run -it --net host --gpus all -v /my_dir:/app -v /localdata/my_data/:/app/serverdata --name ar_run ar_unet:latest

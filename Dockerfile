FROM continuumio/miniconda3
WORKDIR /app

#COPY rf_tf.yml .
#RUN conda env create --name rf_tf --file=rf_tf.yml
#RUN /bin/bash -c "source activate rf_tf && \
#     conda install conda-forge::tensorflow-datasets=4.8.0"

# RUN conda create -n rf_tf python=3.9.20 -y && \
#     /bin/bash -c "source activate rf_tf && \
#     conda install -y numpy tqdm h5py && \
#     conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 && \
#     conda install -y -c conda-forge tensorflow-gpu && \
#     conda install -y -c nvidia cuda-nvcc && \
#     pip install sionna==0.10.0"

# RUN echo "source activate rf_tf && export LD_LIBRARY_PATH=/opt/conda/envs/rf_tf/lib:$LD_LIBRARY_PATH" >> ~/.bashrc

RUN conda create -n rftorch python=3.11.11 -y && \
   /bin/bash -c "source activate rftorch && \
   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia && \
   conda install -y numpy tqdm h5py && \
   conda install -y -c conda-forge tensorflow-cpu=2.15.0 && \
   conda install -y -c conda-forge omegaconf && \
   pip install sionna==0.12.1"
RUN echo "source activate rftorch" >> ~/.bashrc

# docker build -t ariel_wave .
# docker run -it --net host --gpus all -v /home/dsi/arielro1/tmp/rfproj/:/app --name ariel_run1 ariel_wave:latest

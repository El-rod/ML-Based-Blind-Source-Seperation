FROM continuumio/miniconda3
WORKDIR /app

RUN conda create -n rfc_tf python=3.7.13 -y && \
    /bin/bash -c "source activate rfc_tf && \
    conda install -y numpy tqdm h5py && \
    conda install -c nvidia cuda-nvcc && \
    conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 && \
    conda install anaconda::tensorflow-gpu && \
    pip install sionna==0.10.0"

RUN echo "source activate rfc_tf && export LD_LIBRARY_PATH=/opt/conda/envs/rf_tf/lib:$LD_LIBRARY_PATH" >> ~/.bashrc

#RUN conda create -n rfc_torch python=3.7.9 -y && \
#    /bin/bash -c "source activate rfc_torch && \
#    conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch && \
#    conda install -y numpy tqdm h5py && \
#    conda install -y -c conda-forge tensorflow-cpu=2.10.0 && \
#    conda install -y -c conda-forge omegaconf && \
#    pip install sionna==0.12.1"
#RUN echo "source activate rfc_torch" >> ~/.bashrc

# docker build -t ariel_unet .
# docker run -it --net host --gpus all -v /home/dsi/arielro1/tmp/rfproj/:/app --name ariel_run1 ariel_unet:latest
# docker run -it --net host --gpus all -v C:\Users\USER\Documents\GitHub\RFchallenge-ArielProject:/app --name ariel_run1 ariel_unet:latest

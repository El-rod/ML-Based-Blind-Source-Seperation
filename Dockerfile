FROM continuumio/miniconda3
WORKDIR /app

#COPY rfsionna_env.yml .
#RUN conda env create --name rfsionna --file=rfsionna_env.yml

COPY rftorch_env.yml .
RUN conda env create --name rftorch --file=rftorch_env.yml

#RUN conda create -n rftorch python=3.7.9 -y && \
#    /bin/bash -c "source activate rftorch && \
#    conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch && \
#    conda install -y numpy tqdm h5py && \
#    conda install -y -c conda-forge tensorflow=2.10.0 && \
#    conda install -y -c conda-forge omegaconf && \
#    pip install sionna==0.12.1"

# docker build -t conda_test .
# docker run -it --net host --gpus all -v C:\Users\USER\Documents\GitHub\RFchallenge-ArielProject:/app --name to_run1 conda_test:latest

FROM ubuntu:18.04

# ViZdoom dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    bzip2 \
    cmake \
    curl \
    git \
    libboost-all-dev \
    libbz2-dev \
    libfluidsynth-dev \
    libfreetype6-dev \
    libgme-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libopenal-dev \
    libpng-dev \
    libsdl2-dev \
    libsdl2-2.0-0 \
    libgles2-mesa-dev \
    libglu1-mesa-dev \
    libwildmidi-dev \
    libzmq3-dev \
    mesa-utils \
    libgl1-mesa-glx \
    nano \
    nasm \
    pkg-config \
    rsync \
    software-properties-common \
    sudo \
    tar \
    timidity \
    unzip \
    wget \
    zlib1g-dev

RUN apt-get update && apt-get install -y dbus



# Python with pip
RUN apt-get update && apt-get install -y python-dev python python-pip
RUN pip install pip --upgrade


RUN pip --no-cache-dir install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl

# Vizdoom and other pip packages if needed
RUN pip --no-cache-dir install \
    git+https://github.com/mwydmuch/ViZDoom

RUN pip --no-cache-dir install opencv-python termcolor tqdm subprocess32 msgpack-python msgpack-numpy


# Enables X11 sharing and creates user home directory
ENV USER_NAME cig2017
ENV HOME_DIR /home/$USER_NAME

# Replace HOST_UID/HOST_GUID with your user / group id (needed for X11)
ENV HOST_UID 1000
ENV HOST_GID 1000

RUN export uid=${HOST_UID} gid=${HOST_GID} && \
    mkdir -p ${HOME_DIR} && \
    echo "$USER_NAME:x:${uid}:${gid}:$USER_NAME,,,:$HOME_DIR:/bin/bash" >> /etc/passwd && \
    echo "$USER_NAME:x:${uid}:" >> /etc/group && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME && \
    chown ${uid}:${gid} -R ${HOME_DIR}

USER ${USER_NAME}
WORKDIR ${HOME_DIR}


# Copy agent files inside Docker image:
COPY IntelAct_track2 .


### Do not change this ###
COPY *.wad ./
COPY _vizdoom.cfg .
##########################
RUN sudo chown ${HOST_UID}:${HOST_GID} -R *


CMD taskset -c 1 python run_agent.py

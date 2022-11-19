FROM nvidia/cuda:11.7.0-runtime-ubuntu20.04

# Set working directory
ENV WORKDIR=/code
WORKDIR $WORKDIR

RUN useradd -ms /bin/bash admin
RUN chown -R admin:admin $WORKDIR
RUN chmod 755 $WORKDIR


# Install base utilities
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python 3.9 from ppa
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y \
    python3.9 \
    python3-pip

# Link python3.9 to python3 and python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3 & \
    ln -sf /usr/bin/python3 /usr/bin/python & \
    ln -sf /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip


COPY requirements.txt $WORKDIR/requirements.txt

RUN pip install gradio --no-cache-dir
RUN pip install --no-cache-dir --upgrade -r $WORKDIR/requirements.txt

COPY . .

USER admin

EXPOSE 7860

ENTRYPOINT ["python", "app.py", "--path-conf", "config.merged.gpu.yml"]

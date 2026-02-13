FROM --platform=linux/amd64 ubuntu:22.04
VOLUME ["/data"]

ENV DEBIAN_FRONTEND=noninteractive

# Grundsystem + Werkzeuge
RUN apt-get update -y && \
    apt-get install -y wget curl bzip2 bash coreutils openjdk-17-jdk maven python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Miniconda installieren (architekturabhängig)
RUN set -e && \
    arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
        miniconda_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then \
        miniconda_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
        echo "Unsupported architecture: $arch" && exit 1; \
    fi && \
    wget -O /tmp/miniconda.sh ${miniconda_url} && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda config --set always_yes yes --set changeps1 no && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true && \
    /opt/conda/bin/conda update -q conda && \
    /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:$PATH"

# Conda-Environment „venv“ mit allen Paketen erstellen
RUN echo "name: venv\n\
channels:\n\
  - conda-forge\n\
dependencies:\n\
  - matplotlib=3.7.1\n\
  - pandas=1.5.3\n\
  - scipy=1.10.1\n\
  - numpy=1.23.5\n\
  - geopandas=0.12.2\n\
  - numba=0.56.4\n\
  - palettable=3.3.0\n\
  - scikit-learn=1.2.2\n\
  - shapely=2.0.1\n\
  - tqdm=4.65.0\n\
  - pytables=3.7.0\n\
  - xlrd=2.0.1\n\
  - openpyxl=3.1.0\n\
  - pip=23.0.1\n\
  - python=3.10.10\n\
  - py7zr=0.20.8\n\
  - pytest=7.2.2\n\
  - xlwt=1.3.0\n\
  - fiona=1.9.2\n\
  - sqlite=3.42.0\n\
  - pip:\n\
    - synpp==1.5.1" > /tmp/environment.yml && \
    conda env create -f /tmp/environment.yml && \
    rm /tmp/environment.yml && \
    echo "Verifying Conda and Python installation..." && \
    conda --version && \
    /opt/conda/envs/venv/bin/python --version

# Projektdateien
COPY . /root/

SHELL ["/bin/bash", "-c"]
RUN /opt/conda/envs/venv/bin/python --version
# Startkommando mit Conda-Umgebung
#ENTRYPOINT ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate venv && bash docker_run.sh"]
RUN /opt/conda/bin/conda install -n venv -c conda-forge lxml
RUN /opt/conda/bin/conda run -n venv pip install fiona
VOLUME ["/cache"]
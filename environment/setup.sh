#!/bin/bash
set -e

# Define Miniconda3
miniconda_version="4.6.14"
miniconda_url="https://repo.anaconda.com/miniconda/Miniconda3-${miniconda_version}-Linux-x86_64.sh"
miniconda_md5="718259965f234088d785cad1fbd7de03"

python_version="3.6"

jdk_version="8u212"
jdk_url="https://github.com/AdoptOpenJDK/openjdk8-binaries/releases/download/jdk8u212-b03/OpenJDK8U-jdk_x64_linux_hotspot_8u212b03.tar.gz"
jdk_sha256="dd28d6d2cde2b931caf94ac2422a2ad082ea62f0beee3bf7057317c53093de93"

maven_version="3.6.1"
maven_url="http://mirror.easyname.ch/apache/maven/maven-3/${maven_version}/binaries/apache-maven-${maven_version}-bin.tar.gz"
maven_sha512="b4880fb7a3d81edd190a029440cdf17f308621af68475a4fe976296e71ff4a4b546dd6d8a58aaafba334d309cc11e638c52808a4b0e818fc0fd544226d952544"

# Define Python requirements
python_requirements=$(cat <<EOF
tqdm==4.30.0
pyyaml==3.13
pandas==0.24.1
pyproj==1.9.6
geopandas==0.4.0
scikit-learn==0.20.2
requests==2.21.0
rtree==0.8.3
xlrd==1.2.0
jinja2==2.9.5
EOF
)

# Miniconda update script to avoid too long paths in interpreter path
miniconda_update_script=$(cat <<EOF
import sys
import re

with open(sys.argv[1]) as f:
    content = f.read()
    content = re.sub(r'#!(.+)/miniconda3/bin/python', '#!/usr/bin/env python', content)

with open(sys.argv[1], "w+") as f:
    f.write(content)
EOF
)

# I) Ensure the target directory is there
environment_directory=$(realpath "$1")

if [ ! -d ${environment_directory} ]; then
    echo "Creating target directory: ${environment_directory}"
    mkdir -p ${environment_directory}
else
    echo "Target directory already exists: ${environment_directory}"
fi

cd ${environment_directory}

# II) Downloads

## II.1) Download Miniconda
if [ "$(md5sum miniconda.sh)" == "${miniconda_md5}  miniconda.sh" ]; then
    echo "Miniconda 3 ${miniconda_version} already downloaded."
else
    echo "Downloading Miniconda3 ${miniconda_version} ..."
    rm -rf miniconda_installed
    rm -rf python_installed
    curl -o miniconda.sh ${miniconda_url}
fi

## II.2) Download JDK
if [ "$(sha256sum jdk.tar.gz)" == "${jdk_sha256}  jdk.tar.gz" ]; then
    echo "OpenJDK ${jdk_version} already downloaded."
else
    echo "Downloading OpenJDK ${jdk_version} ..."
    rm -rf jdk_installed
    curl -L -o jdk.tar.gz ${jdk_url}
fi

## II.3) Download Maven
if [ "$(sha512sum maven.tar.gz)" == "${maven_sha512}  maven.tar.gz" ]; then
    echo "Maven ${maven_version} already downloaded."
else
    echo "Maven ${maven_version} ..."
    rm -rf maven_installed
    curl -o maven.tar.gz ${maven_url}
fi

# III) Install everything

# III.1) Install Miniconda
if [ -f miniconda_installed ]; then
    echo "Miniconda3 ${miniconda_version} already installed."
else
    echo "Installing Miniconda3 ${miniconda_version} ..."

    rm -rf miniconda3
    sh miniconda.sh -b -u -p miniconda3

    cat <<< "${miniconda_update_script}" > fix_conda.py

    PATH=${environment_directory}/miniconda3/bin:$PATH
    python fix_conda.py miniconda3/bin/conda
    python fix_conda.py miniconda3/bin/conda-env
    conda update -y conda

    touch miniconda_installed
fi

# III.2) Create Python environment
if [ -f python_installed ]; then
    echo "Python environment is already set up."
else
    echo "Setting up Python environment ..."

    cat <<< "${python_requirements}" > requirements.txt
    conda create -p venv python=${python_version} --no-default-packages --channel conda-forge --file requirements.txt -y

    touch python_installed
fi

# III.3) Install OpenJDK
if [ -f jdk_installed ]; then
    echo "OpenJDK ${jdk_version} is already installed."
else
    echo "Installing OpenJDK ${jdk_version} ..."

    mkdir -p jdk
    tar xz -C jdk --strip=1 -f jdk.tar.gz

    touch jdk_installed
fi

# III.4) Install Maven
if [ -f maven_installed ]; then
    echo "Maven ${maven_version} is already installed."
else
    echo "Installing Maven ${maven_version} ..."

    PATH=${environment_directory}/jdk/bin:$PATH
    JAVA_HOME=${environment_directory}/jdk

    mkdir -p maven
    tar xz -C maven --strip=1 -f maven.tar.gz

    touch maven_installed
fi

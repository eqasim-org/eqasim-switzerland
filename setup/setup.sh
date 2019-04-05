#!/bin/bash
set -e

echo "Setting up pipeline"

# define setup paths
ORIG_DIR=$PWD
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SETUP_DIR=${ROOT_DIR}/pipeline_environment
DOWNLOAD_DIR=${SETUP_DIR}/downloads

# make directories
mkdir -p ${SETUP_DIR}
mkdir -p ${DOWNLOAD_DIR}

# download miniconda
CONDA_INSTALLER=${DOWNLOAD_DIR}/miniconda.sh
if [ ! -f ${CONDA_INSTALLER} ]; then
  echo "Downloading miniconda..."
  curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ${CONDA_INSTALLER}
else
  echo "miniconda already downloaded under ${CONDA_INSTALLER}"
fi

# install miniconda
CONDA_DIR=${DOWNLOAD_DIR}/miniconda3
if [ ! -d ${CONDA_DIR} ]
then
  echo "Installing miniconda..."
  sh ${CONDA_INSTALLER} -b -u -p ${CONDA_DIR}
else
  echo "miniconda already installed under ${CONDA_DIR}"
fi

# update conda
echo "Updating conda..."
PATH=${CONDA_DIR}/bin:$PATH
conda update -y conda

# create python environment
echo "Creating python virtual environment..."
PYTHON_VENV_DIR=${SETUP_DIR}/python-venv/
PYTHON_VERSION=3.6
REQUIREMENTS=${ROOT_DIR}/requirements.txt
conda create -p ${PYTHON_VENV_DIR} python=${PYTHON_VERSION} --no-default-packages --channel conda-forge --file ${REQUIREMENTS} -y

# download Oracle JDK
JAVA_TARBALL=jdk-8u201-linux-x64.tar.gz
if [ ! -f ${DOWNLOAD_DIR}/${JAVA_TARBALL} ]
then
  echo "Downloading Oracle JDK..."
  cd ${DOWNLOAD_DIR}
  curl -LOb "oraclelicense=a" https://download.oracle.com/otn-pub/java/jdk/8u201-b09/42970487e3af4f5aa5bca3f542482c60/${JAVA_TARBALL}
else
  echo "Oracle JDK already downloaded under ${DOWNLOAD_DIR}/${JAVA_TARBALL}"
fi

# extract Oracle JDK files
JAVA_DIR=${SETUP_DIR}/jdk1.8.0_201
if [ ! -d ${JAVA_DIR} ]
then
  echo "Unzipping contents of ${JAVA_TARBALL}"
  cd ${DOWNLOAD_DIR}
  tar xzvf ${JAVA_TARBALL} -C ${SETUP_DIR}
else
  echo "Oracle JDK already extracted under ${JAVA_DIR}"
fi

# download maven
MAVEN_TARBALL=apache-maven-3.6.0-bin.tar.gz
if [ ! -f ${DOWNLOAD_DIR}/${MAVEN_TARBALL} ]
then
  echo "Downloading Maven..."
  cd ${DOWNLOAD_DIR}
  curl -O http://mirror.easyname.ch/apache/maven/maven-3/3.6.0/binaries/${MAVEN_TARBALL}
else
  echo "Maven already downloaded under ${DOWNLOAD_DIR}/${MAVEN_TARBALL}"
fi

# extract maven files
MAVEN_DIR=${SETUP_DIR}/apache-maven-3.6.0
if [ ! -d ${MAVEN_DIR} ]
then
  echo "Unzipping contents of ${MAVEN_TARBALL}"
  cd ${DOWNLOAD_DIR}
  tar xzvf ${MAVEN_TARBALL} -C ${SETUP_DIR}
else
  echo "Maven already extracted under ${MAVEN_DIR}"
fi

cd ${ORIG_DIR}

echo "Done!"

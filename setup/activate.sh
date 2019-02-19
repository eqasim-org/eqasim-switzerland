#!/bin/bash

# get directory of bash script
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SETUP_DIR=${ROOT_DIR}/pipeline_environment

echo ${SETUP_DIR}

# add conda to paths
CONDA_DIR=${SETUP_DIR}/downloads/miniconda3/
if [ ! -d ${CONDA_DIR} ]; then
  echo "Could not find conda directory : ${CONDA_DIR}."
  echo "Please run setup.sh script again."
  return
fi
PATH=${CONDA_DIR}/bin:$PATH

# activate python virtual environment
PYTHON_VENV_DIR=${SETUP_DIR}/python-venv/
if [ ! -d ${PYTHON_VENV_DIR} ]; then
  echo "Could not find virtual environment : ${PYTHON_VENV_DIR}."
  echo "Please run setup.sh script again."
  return
fi
source activate ${PYTHON_VENV_DIR}

# add java to path
echo "Adding java to PATH..."
JAVA_DIR=${SETUP_DIR}/jdk1.8.0_201/
if [ ! -d ${JAVA_DIR} ]; then
  echo "Could not find java directory : ${JAVA_DIR}."
  echo "Please run setup.sh script again."
  return
fi
PATH=${JAVA_DIR}/bin:$PATH

# add mvn to path
echo "Adding maven to PATH..."
MAVEN_DIR=${SETUP_DIR}/apache-maven-3.6.0/
if [ ! -d ${MAVEN_DIR} ]; then
  echo "Could not find java directory : ${MAVEN_DIR}."
  echo "Please run setup.sh script again."
  return
fi
PATH=${MAVEN_DIR}/bin:$PATH

echo "Done!"

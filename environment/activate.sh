#!/bin/bash

environment_directory=$(realpath "$1")

if [ ! -f ${environment_directory}/miniconda_installed ]; then
    echo "Miniconda is not installed properly."
    exit 1
else
    PATH=${environment_directory}/miniconda3/bin:$PATH

    echo "Testing Miniconda ..."
    conda -V
fi

if [ ! -f ${environment_directory}/python_installed ]; then
    echo "Python environment is not installed properly."
    exit 1
else
    source activate ${environment_directory}/venv

    echo "Testing Python ..."
    python3 --version
fi

if [ ! -f ${environment_directory}/jdk_installed ]; then
    echo "OpenJDK is not installed properly."
    exit 1
else
    PATH=${environment_directory}/jdk/bin:$PATH
    JAVA_HOME=${environment_directory}/jdk

    echo "Testing OpenJDK ..."
    java -version
    javac -version
fi

if [ ! -f ${environment_directory}/maven_installed ]; then
    echo "Maven is not installed properly."
    exit 1
else
    PATH=${environment_directory}/maven/bin:$PATH

    echo "Testing Maven ..."
    mvn -version
fi

echo "Environment is set up."

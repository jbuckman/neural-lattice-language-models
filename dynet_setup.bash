#!/bin/bash
# Code to easily set up DyNet on a new GCE instance w/ CUDA. Mostly replicated from DyNet tutorial.

apt-get -y update
apt-get install -y python-pip build-essential cmake mercurial

pip install --upgrade pip
pip install cython

mkdir -p dynet-base
cd dynet-base

CUDA_VERSION_MAJOR="8" CUDA_VERSION_MINOR="0"
CUDA_PKG_LONGVERSION="${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}.61-1"
CUDA_PKG_VERSION="${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}"
CUDA_REPO_PKG=cuda-repo-ubuntu1404_${CUDA_PKG_LONGVERSION}_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG
dpkg -i $CUDA_REPO_PKG
apt-get -y update
apt-get install -y --no-install-recommends cuda-drivers cuda-core-$CUDA_PKG_VERSION cuda-cudart-dev-$CUDA_PKG_VERSION cuda-cublas-dev-$CUDA_PKG_VERSION cuda-curand-dev-$CUDA_PKG_VERSION
ln -s /usr/local/cuda-${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} /usr/local/cuda

git clone https://github.com/clab/dynet.git
hg clone https://bitbucket.org/eigen/eigen -r 699b659
cd dynet
mkdir -p build
cd build

cmake .. -DEIGEN3_INCLUDE_DIR=../../eigen -DPYTHON=`which python` -DBACKEND=cuda

make -j 16
cd python
python ../../setup.py build --build-dir=.. --skip-build install

#!/bin/bash
#
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

WRKSPC=$(pwd)
# everything will be installed in $WRKSPC

ENV_NAME="venv"
# this is the name of your python venv, change if needed


cd $WRKSPC || exit 1

module purge 
module load PrgEnv-amd
module load rocm
module load cray-mpich
module load miniforge3

echo -e "${RED}Creating Python Environment in $WRKSPC:${GREEN}"
rm -rf $WRKSPC/$ENV_NAME
python3.10 -m venv $WRKSPC/$ENV_NAME
source $WRKSPC/$ENV_NAME/bin/activate
pip install --upgrade pip

pip install "numpy<2"

pip uninstall torch -y
pip install torch==1.12.0+rocm4.5 torchvision==0.13.0+rocm4.5 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/rocm4.5
pip install dgl==0.9.1
pip install ogb==1.3.5
pip install numpy scipy scikit-learn tqdm
pip install -r requirements.txt --no-deps

cd $WRKSPC

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
python -m venv $WRKSPC/$ENV_NAME
source $WRKSPC/$ENV_NAME/bin/activate
pip install --upgrade pip

pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
pip install scipy

pip install -r requirements.txt --no-deps

pip install https://pypi.amd.com/rocm-7.0.0/packages/amd-dgl/amd_dgl-2.4.0+amd0.torch2.8.0.rocm7.0.0.git64359f59.ubuntu24.4-cp312-cp312-linux_x86_64.whl

cd dgl

cd $WRKSPC

python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
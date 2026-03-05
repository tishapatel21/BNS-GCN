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

pip install torch==2.3.1+rocm5.7 torchvision==0.15.2+rocm5.7 --index-url https://download.pytorch.org/whl/rocm5.7

pip install torchdata==0.7.0
pip install pandas>=1.5 scikit-learn>=1.2 six>=1.16 outdated>=0.2.2 tqdm request scipy
pip install ogb==1.3.6
pip install dgl==2.1.0

pip install -r requirements.txt --no-deps

cd $WRKSPC

python -c "import torch; import dgl; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), dgl.version)"
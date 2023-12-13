set -e

# Setup expected paths
STRIPS_HGN_DIR="$HOME/strips-hgn"
SRC_DIR="$STRIPS_HGN_DIR/src"
FD_DIR="${SRC_DIR}/fast_downward"

VIRTUAL_ENV="pytorch_latest_p36"

#############################
## Setup Lab ##
#############################


cd $HOME
sudo apt install g++ make flex bison

echo "Cloning Val from Gitlab."
git clone https://github.com/KCL-Planning/VAL.git
cd VAL
git checkout a5565396007eee73ac36527fbf904142b3077c74
make clean  # Remove old binaries.
sed -i 's/-Werror //g' Makefile  # Ignore warnings.
make

# Add relevant environment variables to .bashrc
cd $HOME
echo "export FD_HGN=\"$HOME/fd-hypernet\"" >> .bashrc
echo "export STRIPS_HGN_NEW=\"${STRIPS_HGN_DIR}\"" >> .bashrc
echo "export PYTHONPATH=\"${SRC_DIR}\"" >> .bashrc
echo "export MKL_THREADING_LAYER=GNU" >> .bashrc
echo "export PATH=$PATH:${HOME}/VAL/"

# For faster terminal navigation
echo "bind '\"\e[A\":history-search-backward'" >> .bashrc
echo "bind '\"\e[B\":history-search-forward'" >> .bashrc

# Activate PyTorch virtual env
echo "Activating ${VIRTUAL_ENV} conda environment"
source activate $VIRTUAL_ENV

# Install pybind11
conda install pybind11

# Install lab
pip install lab

First check gcc/g++ version by:

```
gcc --version
g++ --version
```

If it does not return with `gcc/g++ (Ubuntu 11.4...)...` , do the following gcc/g++ update:

```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update

sudo apt install gcc-11 g++-11

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110

sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

Once it returns the corrct gcc/g++ version, follow the commands in below for nerfstudio installation:

```
conda create --name nerfstudio -y python=3.10.0
conda activate nerfstudio
python -m pip install --upgrade pip

pip uninstall torch torchvision functorch tinycudann

pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit

pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install nerfstudio
```

Other commands to check the installed versions of torch and torchvision.

```
pip show torch
pip show torchvision
```
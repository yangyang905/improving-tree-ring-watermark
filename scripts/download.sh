# /bin/bash

wget https://dl.fbaipublicfiles.com/wmar/syncseal/paper/syncmodel.jit.pt -P checkpoints
wget https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P checkpoints/
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt -P pre_trained

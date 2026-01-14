# README
下载 SyncSeal 模型：
```bash
wget -O syncmodel.jit.pt https://dl.fbaipublicfiles.com/wmar/syncseal/paper/syncmodel.jit.pt
```
目前只做了 imagenet 256 × 256 模型上的代码修改（run_tree_ring_watermark_imagenet.py），运行方式见 scripts/tree_ring_imagenet.sh
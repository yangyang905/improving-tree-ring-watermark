from .base import BaseSync

import torch


class SyncSeal(BaseSync):
    def __init__(self, syncpath, device):
        self.model = torch.jit.load(syncpath, map_location=device).eval()
        self.device = device

    # transfer to SYNC space, [-1, 1] -> [0, 1]
    def normalize(self, imgs):
        return (imgs + 1.0) / 2.0

    # transfer from SYNC space, [0, 1]-> [-1, 1]
    def unnormalize(self, imgs):
        imgs = imgs * 2.0 - 1.0
        return imgs.clamp(-1, 1)


    # imgs: [b, 3, 256, 256] in [-1, 1] -> return same
    def add_sync(self, imgs, return_masks=False):
        assert return_masks == False, "return_masks not supported for SyncSeal"
        orig_device = imgs.device
        imgs = self.normalize(imgs).to(self.device)
        with torch.no_grad():
            imgs_w = self.model.embed(imgs)["imgs_w"]
        ret = self.unnormalize(imgs_w).to(orig_device)
        return ret

    # imgs: [b, 3, 256, 256] in [-1, 1] -> return same
    def remove_sync(self, imgs, return_info=False):
        assert return_info == False, "return_masks not supported for SyncSeal"
        orig_device = imgs.device
        orig_size = imgs.shape[-2], imgs.shape[-1]
        imgs = self.normalize(imgs).to(self.device)

        with torch.no_grad():
            det = self.model.detect(imgs)
            pred_pts = det['preds_pts']  # Bx8 normalized [-1,1]
            imgs_unwarped = self.model.unwarp(imgs, pred_pts, orig_size) 

        ret = self.unnormalize(imgs_unwarped).to(orig_device)
        return ret

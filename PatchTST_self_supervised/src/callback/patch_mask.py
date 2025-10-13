
import torch
from torch import nn

from .core import Callback

# Cell
class PatchCB(Callback):

    def __init__(self, patch_len, stride ):
        """
        Callback used to perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride

    def before_forward(self): self.set_patch()
       
    def set_patch(self):
        """
        take xb from learner and convert to patch: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb: [bs x seq_len x n_vars]
        # learner get the transformed input
        self.learner.xb = xb_patch                              # xb_patch: [bs x num_patch x n_vars x patch_len]           


# class PatchMaskCB(Callback):
#     def __init__(self, patch_len, stride, mask_ratio, use_gaussian_noise, noise_std,
#                         mask_when_pred:bool=False):
#         """
#         Callback used to perform the pretext task of reconstruct the original data after a binary mask has been applied.
#         Args:
#             patch_len:        patch length
#             stride:           stride
#             mask_ratio:       mask ratio
#         """
#         self.patch_len = patch_len
#         self.stride = stride
#         self.mask_ratio = mask_ratio
#         self.use_gaussian_noise = use_gaussian_noise
#         self.noise_std = noise_std

#     def before_fit(self):
#         # overwrite the predefined loss function
#         self.learner.loss_func = self._loss        
#         device = self.learner.device       
 
#     def before_forward(self): self.patch_masking()
        
#     def patch_masking(self):
#         """
#         xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
#         """
#         xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb_patch: [bs x num_patch x n_vars x patch_len]
#         xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio, use_gaussian_noise=self.use_gaussian_noise, noise_std=self.noise_std)   # xb_mask: [bs x num_patch x n_vars x patch_len]
#         self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
#         self.learner.xb = xb_mask       # learner.xb: masked 4D tensor    
#         self.learner.yb = xb_patch      # learner.yb: non-masked 4d tensor
 
#     def _loss(self, preds, target):        
#         """
#         preds:   [bs x num_patch x n_vars x patch_len]
#         targets: [bs x num_patch x n_vars x patch_len] 
#         """
#         loss = (preds - target) ** 2
#         loss = loss.mean(dim=-1)
#         loss = (loss * self.mask).sum() / self.mask.sum()
#         return loss


class PatchMaskCB(Callback):
    def __init__(self, patch_len, stride, mask_ratio, use_gaussian_noise, noise_std,
                 use_mask_token=False, mask_when_pred:bool=False):
        """
        Callback used to perform the pretext task of reconstruct the original data after a binary mask has been applied.
        Args:
            patch_len:          patch length
            stride:             stride
            mask_ratio:         mask ratio
            use_gaussian_noise: whether to use Gaussian noise for input-level masking
            noise_std:          standard deviation of Gaussian noise
            use_mask_token:     whether to use embedding-level mask token
        """
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio
        self.use_gaussian_noise = use_gaussian_noise
        self.noise_std = noise_std
        self.use_mask_token = use_mask_token

    def before_fit(self):
        # overwrite the predefined loss function
        self.learner.loss_func = self._loss        
        device = self.learner.device       
 
    def before_forward(self): 
        self.patch_masking()
        
    def patch_masking(self):
        """
        xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb_patch: [bs x num_patch x n_vars x patch_len]
        xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio, 
                                                   use_mask_token=self.use_mask_token,
                                                   use_gaussian_noise=self.use_gaussian_noise, 
                                                   noise_std=self.noise_std)   # xb_mask: [bs x num_patch x n_vars x patch_len]
        self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
        
        # Store the mask for the model if using mask token
        if self.use_mask_token:
            self.learner.mask = self.mask  # Pass mask to learner so it can pass to model
        
        self.learner.xb = xb_mask       # learner.xb: masked 4D tensor (or original if use_mask_token)
        self.learner.yb = xb_patch      # learner.yb: non-masked 4d tensor
 
    def _loss(self, preds, target):        
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len] 
        """
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.mask).sum() / self.mask.sum()
        return loss


def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
    tgt_len = patch_len  + stride*(num_patch-1)
    s_begin = seq_len - tgt_len
        
    xb = xb[:, s_begin:, :]                                                    # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)                 # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patch


class Patch(nn.Module):
    def __init__(self,seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
        tgt_len = patch_len  + stride*(self.num_patch-1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        """
        x: [bs x seq_len x n_vars]
        """
        x = x[:, self.s_begin:, :]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)                 # xb: [bs x num_patch x n_vars x patch_len]
        return x


# def random_masking(xb, mask_ratio, use_gaussian_noise, noise_std):
#     # xb: [bs x num_patch x n_vars x patch_len]
#     bs, L, nvars, D = xb.shape
#     x = xb.clone()
    
#     len_keep = int(L * (1 - mask_ratio))
        
#     noise = torch.rand(bs, L, nvars, device=xb.device)  # noise in [0, 1], bs x L x nvars
        
#     # sort noise for each sample
#     ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#     ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L x nvars]

#     # keep the first subset
#     ids_keep = ids_shuffle[:, :len_keep, :]  # ids_keep: [bs x len_keep x nvars]         
#     x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))  # x_kept: [bs x len_keep x nvars x patch_len]
   
#     # removed x - MODIFIED PART
#     if use_gaussian_noise:
#         # print(f"[INFO] Adding gaussian noise with std {noise_std}")
#         # Add Gaussian noise instead of zeros
#         x_removed = torch.randn(bs, L-len_keep, nvars, D, device=xb.device) * noise_std
#     else:
#         # Original: set to zeros
#         # print(f"[INFO] Using zero masking")
#         x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)
    
#     x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len]

#     # combine the kept part and the removed one
#     x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D))  # x_masked: [bs x num_patch x nvars x patch_len]

#     # generate the binary mask: 0 is keep, 1 is remove
#     mask = torch.ones([bs, L, nvars], device=x.device)  # mask: [bs x num_patch x nvars]
#     mask[:, :len_keep, :] = 0
#     # unshuffle to get the binary mask
#     mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch x nvars]
#     return x_masked, x_kept, mask, ids_restore


# def random_masking(xb, mask_ratio, use_gaussian_noise, noise_std):
#     # xb: [bs x num_patch x n_vars x patch_len]
#     bs, L, nvars, D = xb.shape
#     x = xb.clone()
    
#     len_keep = int(L * (1 - mask_ratio))
        
#     noise = torch.rand(bs, L, nvars, device=xb.device)  # noise in [0, 1], bs x L x nvars
        
#     # sort noise for each sample
#     ids_shuffle = torch.argsort(noise, dim=1)
#     ids_restore = torch.argsort(ids_shuffle, dim=1)

#     # keep the first subset
#     ids_keep = ids_shuffle[:, :len_keep, :]
#     x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))
   
#     # Get the patches to be masked
#     ids_remove = ids_shuffle[:, len_keep:, :]
#     x_to_mask = torch.gather(x, dim=1, index=ids_remove.unsqueeze(-1).repeat(1, 1, 1, D))
    
#     # OPTION 1: Add Gaussian noise to the original patch values
#     if use_gaussian_noise:
#         # print(f"[INFO] Adding gaussian noise with std {noise_std}")
#         gaussian_noise = torch.randn(bs, L-len_keep, nvars, D, device=xb.device) * noise_std
#         x_removed = x_to_mask + gaussian_noise  # Add noise to original values
#     else:
#         # Original: set to zeros
#         x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)
    
#     x_ = torch.cat([x_kept, x_removed], dim=1)

#     # combine the kept part and the masked one
#     x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D))

#     # generate the binary mask: 0 is keep, 1 is remove
#     mask = torch.ones([bs, L, nvars], device=x.device)
#     mask[:, :len_keep, :] = 0
#     mask = torch.gather(mask, dim=1, index=ids_restore)
#     return x_masked, x_kept, mask, ids_restore

def random_masking(xb, mask_ratio, use_mask_token=False, use_gaussian_noise=False, noise_std=0.1):
    """
    xb: [bs x num_patch x n_vars x patch_len]
    mask_ratio: ratio of patches to mask
    use_mask_token: if True, return mask for embedding-level masking; 
                    if False, modify patches directly (input-level masking)
    use_gaussian_noise: if True, add/replace with Gaussian noise (only for input-level masking)
    noise_std: standard deviation of Gaussian noise
    
    Returns:
        x_masked: modified patches or original depending on use_mask_token
        x_kept: kept patches for reconstruction target
        mask: binary mask [bs x num_patch x nvars], 1=masked, 0=keep
        ids_restore: indices to restore original order
    """
    bs, L, nvars, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, nvars, device=xb.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))
   
    # Generate the binary mask: 0 is keep, 1 is masked
    mask = torch.ones([bs, L, nvars], device=x.device)
    mask[:, :len_keep, :] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch x nvars]
    
    if use_mask_token:
        # Embedding-level masking: return original patches, let encoder apply mask token
        x_masked = xb
    else:
        # Input-level masking: modify patches directly
        if use_gaussian_noise:
            # Get the patches to be masked
            ids_remove = ids_shuffle[:, len_keep:, :]
            x_to_mask = torch.gather(x, dim=1, index=ids_remove.unsqueeze(-1).repeat(1, 1, 1, D))
            
            # Add Gaussian noise to the original patches (additive noise)
            gaussian_noise = torch.randn(bs, L-len_keep, nvars, D, device=xb.device) * noise_std
            x_removed = x_to_mask + gaussian_noise
        else:
            # Zero masking (original approach)
            x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)
        
        # Combine kept and masked patches
        x_ = torch.cat([x_kept, x_removed], dim=1)
        x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D))
    
    return x_masked, x_kept, mask, ids_restore


def random_masking_3D(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    bs, L, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]                                                 # ids_keep: [bs x len_keep]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))        # x_kept: [bs x len_keep x dim]
   
    # removed x
    x_removed = torch.zeros(bs, L-len_keep, D, device=xb.device)                        # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D))    # x_masked: [bs x num_patch x dim]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)                                          # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch]
    return x_masked, x_kept, mask, ids_restore


if __name__ == "__main__":
    bs, L, nvars, D = 2,20,4,5
    xb = torch.randn(bs, L, nvars, D)
    xb_mask, mask, ids_restore = create_mask(xb, mask_ratio=0.5)
    breakpoint()



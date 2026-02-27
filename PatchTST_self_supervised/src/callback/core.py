
__all__ = ['Callback', 'SetupLearnerCB', 'GetPredictionsCB', 'GetTestCB' ]


""" 
Callback lists:
    > before_fit
        - before_epoch
            + before_epoch_train                
                ~ before_batch_train
                ~ after_batch_train                
            + after_epoch_train

            + before_epoch_valid                
                ~ before_batch_valid
                ~ after_batch_valid                
            + after_epoch_valid
        - after_epoch
    > after_fit

    - before_predict        
        ~ before_batch_predict
        ~ after_batch_predict          
    - after_predict

"""

from ..basics import *
import torch

DTYPE = torch.float32

class Callback(GetAttr): 
    _default='learner'


class SetupLearnerCB(Callback): 
    def __init__(self):        
        self.device = default_device(use_cuda=True)

    def before_batch_train(self): self._to_device()
    def before_batch_valid(self): self._to_device()
    def before_batch_predict(self): self._to_device()
    def before_batch_test(self): self._to_device()

    def _to_device(self):
        batch = to_device(self.batch, self.device)        
        if self.n_inp > 1: xb, yb = batch
        else: xb, yb = batch, None        
        self.learner.batch = xb, yb
        
    def before_fit(self): 
        "Set model to cuda before training"                
        self.learner.model.to(self.device)
        self.learner.device = self.device                        


class GetPredictionsCB(Callback):
    def __init__(self):
        super().__init__()

    def before_predict(self):
        self.preds = []        
    
    def after_batch_predict(self):        
        # append the prediction after each forward batch           
        self.preds.append(self.pred)

    def after_predict(self):           
        self.preds = torch.concat(self.preds)#.detach().cpu().numpy()

         
class GetTestCB(Callback):
    def __init__(self, save_case_study=False, var_idx_mapping=None, batch_size=None, save_path=None):
        super().__init__()
        self.save_case_study = save_case_study
        self.var_idx_mapping = var_idx_mapping
        self.batch_size = batch_size
        self.save_path = save_path
        self.first_batch_saved = False

    def before_test(self):
        self.preds, self.targets = [], []
        self.attentions = []
    
    def after_batch_test(self):
        print(f"DEBUG: after_batch_test called")
        print(f"DEBUG: save_case_study = {self.save_case_study}")
        print(f"DEBUG: first_batch_saved = {self.first_batch_saved}")
        print(f"DEBUG: has attn = {hasattr(self.learn, 'attn')}")        
        # append the prediction after each forward batch           
        self.preds.append(self.pred)
        self.targets.append(self.yb)
        
        # Collect attention if available
        if hasattr(self.learn, 'attn') and self.learn.attn is not None:
            self.attentions.append(self.learn.attn)
            
            # Save first batch for case study
            if self.save_case_study and not self.first_batch_saved and self.var_idx_mapping:
                import os
                import numpy as np
                
                folder_path = self.save_path + 'test_results/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                batch_x = self.learn.xb  # Context
                batch_y = self.learn.yb  # Target
                pred = self.pred         # Predictions
                attn = self.learn.attn   # Attention
                
                for series_num in self.var_idx_mapping.values():
                    sample_position = series_num * self.batch_size
                    
                    # Save attention
                    np.save(folder_path + f'attention_series{series_num}.npy',
                           attn[:, sample_position:sample_position+1].detach().cpu().numpy())
                    
                    # Save context (need to extract from patches)
                    # Note: batch_x might be in patch format [bs, num_patch, nvars, patch_len]
                    # We need the original context - this depends on your data format
                    
                    # Save target
                    np.save(folder_path + f'target_series{series_num}.npy',
                           batch_y[0:1, :, series_num:series_num+1].detach().cpu().numpy())
                    
                    # Save prediction
                    np.save(folder_path + f'prediction_series{series_num}.npy',
                           pred[0:1, :, series_num:series_num+1].detach().cpu().numpy())
                
                self.first_batch_saved = True
                print(f"Saved case study data to {folder_path}")

    def after_test(self):           
        self.preds = torch.concat(self.preds)
        self.targets = torch.concat(self.targets)
        
        # Stack attentions if collected
        if self.attentions:
            self.attentions = torch.cat(self.attentions, dim=1)
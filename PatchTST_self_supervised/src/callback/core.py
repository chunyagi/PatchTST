
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
    def __init__(self, setting, store_attn=False):
        super().__init__()
        self.setting = setting
        self.store_attn = store_attn
        self.first_batch_saved = False

    def before_test(self):
        self.preds, self.targets = [], []
    
    def after_batch_test(self):        
        self.preds.append(self.pred)
        self.targets.append(self.yb)
        
        if self.store_attn and not self.first_batch_saved and hasattr(self.learner, 'attn'):
            import os
            import numpy as np
            
            var_idx_mapping = {
                0: 11, 1: 25, 2: 81, 3: 4, 4: 24, 5: 27, 
                6: 152, 7: 154, 8: 237, 9: 238, 10: 206, 
                11: 202, 12: 37, 13: 8, 14: 113, 15: 114
            }
            
            folder_path = './test_results/' + self.setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            attn = self.learner.attn
            batch_x = self.learner.xb
            batch_y = self.learner.yb
            pred = self.pred
            
            for series_num in var_idx_mapping.values():
                np.save(folder_path + f'attention_series{series_num}.npy',
                      attn[:, series_num:series_num+1].detach().cpu().numpy())
                np.save(folder_path + f'context_series{series_num}.npy',
                      batch_x[0:1, :, series_num:series_num+1].detach().cpu().numpy())
                np.save(folder_path + f'target_series{series_num}.npy',
                      batch_y[0:1, :, series_num:series_num+1].detach().cpu().numpy())
                np.save(folder_path + f'prediction_series{series_num}.npy',
                      pred[0:1, :, series_num:series_num+1].detach().cpu().numpy())
            
            self.first_batch_saved = True
            print(f"Saved case study data to {folder_path}")
    
    def after_test(self):           
        self.preds = torch.concat(self.preds)
        self.targets = torch.concat(self.targets)
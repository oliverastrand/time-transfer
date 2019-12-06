import torch
import torch.nn as nn

class DumbEncoder(nn.Module):
        
        def __init__(self):
            super().__init__()
                            
            self.enc = nn.Linear(10*10*3, 5)
                                            
        def forward(self, x):
            x = x.view(-1, 10*10*3)
            x = self.enc(x)
            return x
                                                                                
                                                                                
                                                                                
class DumbDecoder(nn.Module):
                                                                                
    def __init__(self):
                
        super().__init__()
                                                                                                
        self.dec = nn.Linear(6, 10*10*3)
        
    def forward(self, x):
        x = self.dec(x)
        return x.view(-1, 3, 10, 10)


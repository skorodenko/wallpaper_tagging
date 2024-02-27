import torch
import lightning as lg


class LQP(lg.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential([
            torch.nn.Linear(2048, 512),
            torch.nn.Linear(512, 256),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        ])
        
    def forward(self, x):
        x = self.fc(x)
        return x
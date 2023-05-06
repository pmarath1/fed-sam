import copy
import torch
from torchcontrib.optim import SWA
from .client import Client

class SWAClient(Client):
    def __init__(self, seed, client_id, lr, weight_decay, batch_size, momentum, train_data, eval_data, model, swa_start=5, swa_freq=5, swa_lr=0.005, device=None,
                 num_workers=0, run=None, mixup=False, mixup_alpha=1.0):
        super().__init__(seed, client_id, lr, weight_decay, batch_size, momentum, train_data, eval_data, model, device,
                         num_workers, run, mixup, mixup_alpha)
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr

    def train(self, num_epochs=1, batch_size=10, minibatch=None):
        num_train_samples, update = super().train(num_epochs, batch_size, minibatch)
        return num_train_samples, update

    def run_epoch(self, optimizer, criterion):
        running_loss = 0.0
        i = 0
        for inputs, targets in self.trainloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            with torch.no_grad():
                running_loss += loss.item()

            i += 1

        if self.swa_start and self.swa_freq and self.swa_lr:
            if (i + 1) == self.swa_start or (i + 1 - self.swa_start) % self.swa_freq == 0:
                optimizer.update_swa()

            if (i + 1) == self.swa_start:
                optimizer.swap_swa_sgd()
                optimizer.lr = self.swa_lr

        if i == 0:
            print("Not running epoch", self.id)
            return 0

        return running_loss / i

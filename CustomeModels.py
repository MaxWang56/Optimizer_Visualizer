import torch.nn as nn
import torch.nn.functional as F


class ModelThreeLinearDropout(nn.Module):
    def __init__(self):
        super(ModelThreeLinearDropout, self).__init__()
        self.fc1 = nn.Linear(784, 160)
        self.fc2 = nn.Linear(160, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, 1)

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import time 


class BCINet(nn.Module):
    def __init__(self, feature_dim, hidden_layer_dim, target_dim):
        super(BCINet, self).__init__()
        self.forward1 = nn.Linear(feature_dim, hidden_layer_dim)
        self.forward2 = nn.Linear(hidden_layer_dim, target_dim)

    def forward_pass(self, x):
        x = torch.sigmoid(self.forward1(x))
        x = F.softmax(self.forward2(x), dim=1)
        return x


left_tensor = torch.load('left_tensor')
right_tensor = torch.load('right_tensor')
stop_tensor = torch.load('stop_tensor')

print(left_tensor.shape, right_tensor.shape, stop_tensor.shape)
time.sleep(20)

movements = {'left': 0, 'right': 1, 'stop': 2}
labels = torch.load('tensor_labels')

label = labels[-1]
EPOCHS = 500

target_dim = 18
hidden_dim = 200
feature_dim = 19

net = BCINet(feature_dim, hidden_dim, target_dim)
optimizer = optim.SGD(net.parameters(), lr=0.75)
criterion = nn.CrossEntropyLoss()

for e in range(EPOCHS):
    running_loss = 0.
    for j in rl:
        net.zero_grad()
        j = j.view([1, -1])
        rl_f = net.forward_pass(j.float())
        loss = criterion(rl_f, label)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('Epoch {} / {} Loss = {}'.format(e + 1, EPOCHS, running_loss / len(rl)))


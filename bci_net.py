import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print('using device:', device)

class BCINet(nn.Module):
    def __init__(self, feature_dim, hidden_layer_dim, target_dim):
        super(BCINet, self).__init__()
        self.forward1 = nn.Linear(feature_dim, hidden_layer_dim)
        self.forward2 = nn.Linear(hidden_layer_dim, target_dim)

    def forward_pass(self, x):
        x = torch.sigmoid(self.forward1(x))
        x = F.softmax(self.forward2(x), dim=1)
        return x


def get_samples_features(labels, *tensors):
    train_data = zip([left_tensor, right_tensor, stop_tensor], labels)
    features = left_tensor[0].shape[0]
    total_samples = 0
    for t in train_data:
        total_samples += t[0].shape[0]
    return total_samples, features


left_tensor = torch.load('left_tensor')
right_tensor = torch.load('right_tensor')
stop_tensor = torch.load('stop_tensor')

movements = {'left': 0, 'right': 1, 'stop': 2}
ix_to_movements = dict([(v, k) for k, v in movements.items()])

labels = torch.load('tensor_labels')

print('inferring total samples...')
total_samples, features = get_samples_features(labels, left_tensor, right_tensor, stop_tensor)
print('total samples', total_samples)


test_slice = 2500

train_left_tensor = left_tensor[:-test_slice]
train_right_tensor = right_tensor[:-test_slice]
train_stop_tensor = stop_tensor[:-test_slice]

test_left_tensor = left_tensor[-test_slice:]
test_right_tensor = right_tensor[-test_slice:]
test_stop_tensor = stop_tensor[-test_slice:]

EPOCHS = 50

target_dim = len(labels)
hidden_dim = 50
feature_dim = features

net = BCINet(feature_dim, hidden_dim, target_dim)
optimizer = optim.SGD(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

directions = [train_left_tensor, train_right_tensor, train_stop_tensor]

print('begin training...')

loss_tracker = []
for e in range(EPOCHS):
    running_loss = 0.
    for d in range(len(directions)):
        direction = directions[d]
        label = labels[d]
        for j in direction:
            net.zero_grad()
            j = j.view([1, -1])
            forward = net.forward_pass(j.float())
            loss = criterion(forward, label)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / total_samples
    loss_tracker.append(epoch_loss)
    
    print('Epoch {} / {} Loss = {}'.format(e + 1, EPOCHS, epoch_loss))


accuracy = 0
test_directions = [test_left_tensor, test_right_tensor, test_stop_tensor]
for d in range(len(test_directions)):
    direction = test_directions[d]
    label = labels[d]
    for j in direction:
        net.zero_grad()
        j = j.view([1, -1])
        forward = net.forward_pass(j.float())
        pred = torch.argmax(forward).item()
        true = label.item()
        if pred == true:
            accuracy += 1


accuracy = (accuracy / (len(movements) * test_slice)) * 100
print('Accuracy on test set = {}%'.format(accuracy))

plot = False
if plot:
    plt.plot(loss_tracker)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

import scipy.io as sio 
import os
import torch

train_files = [f for f in os.listdir() if os.path.splitext(f)[1] == '.mat']
print("{} '.mat' files found".format(len(train_files)))
print('files found:', train_files)

left = []
right = []
stop = []

for t in train_files:
    print('Fetching data from {}'.format(t))
    loader = sio.loadmat(t)
    for keys, values in loader.items():
        if keys == 'LeftBackward1':
            data = loader[keys]
            left.append(data)
        if keys == 'LeftBackward2':
            data = loader[keys]
            left.append(data)
        if keys == 'LeftBackward3':
            data = loader[keys]
            left.append(data)
        if keys == 'LeftBackwardImagined':
            data = loader[keys]
            left.append(data)
        if keys == 'LeftForward1':
            data = loader[keys]
            left.append(data)
        if keys == 'LeftForward2':
            data = loader[keys]
            left.append(data)
        if keys == 'LeftForward3':
            data = loader[keys]
            left.append(data)
        if keys == 'LeftForwardImagined':
            data = loader[keys]
            left.append(data)
        if keys == 'LeftLeg':
            data = loader[keys]
            stop.append(data)
        if keys == 'RightBackward1':
            data = loader[keys]
            right.append(data)
        if keys == 'RightBackward2':
            data = loader[keys]
            right.append(data)
        if keys == 'RightBackward3':
            data = loader[keys]
            right.append(data)
        if keys == 'RightBackwardImagined':
            data = loader[keys]
            right.append(data)
        if keys == 'RightForward1':
            data = loader[keys]
            right.append(data)
        if keys == 'RightForward2':
            data = loader[keys]
            right.append(data)
        if keys == 'RightForward3':
            data = loader[keys]
            right.append(data)
        if keys == 'RightForwardImagined':
            data = loader[keys]
            right.append(data)
        if keys == 'RightLeg':
            data = loader[keys]
            stop.append(data)
        

labels = ['left', 'right', 'stop']
tensor_labels = [torch.tensor([j]) for j in range(len(labels))]
print('File processed')

left_tensor = torch.tensor(left[0])
right_tensor = torch.tensor(right[0])
stop_tensor = torch.tensor(stop[0])

torch.save(tensor_labels, 'tensor_labels')
torch.save(left_tensor, 'left_tensor')
torch.save(right_tensor, 'right_tensor')
torch.save(stop_tensor, 'stop_tensor')


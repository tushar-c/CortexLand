import torch 
import torch.nn as nn
import torch.optim as optim


class ModelTrain(object):
    def __init__(self, model, optimizer, criterion, X, y, test_x, test_y, epochs):
        self.model = model 
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        
        self.train_data = [(X[j], y[j]) for j in range(len(X))]
        self.test_data = [(test_x[j], test_y[j]) for j in range(len(test_x))]

        self.train_loss_tracker = []


    def train(self):
        for e in range(self.epochs):
            running_loss = 0.
            for feature, label in self.train_data:
                self.model.zero_grad()
                forward_pass = self.model.forward(feature)
                loss = self.criterion(forward_pass, label)
                running_loss += float(loss)
                loss.backward()
                self.optimizer.step()
            self.train_loss_tracker.append(running_loss)
            print('Epoch {} / {} ; Loss = {}'.format(e + 1, self.epochs, running_loss))
        return self.train_loss_tracker

    def test(self):
        accuracy = 0
        sample_counter = 0
        for feature, label in self.test_data:
            sample_counter += 1
            forward_pass = self.model.forward(feature)
            pred = torch.argmax(forward_pass)
            if pred.item() == label.item():
                accuracy += 1
        return accuracy / sample_counter

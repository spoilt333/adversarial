import torch
from torch import utils
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

as_MNIST_transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
           ])

basic_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
           ])

def mnist(batch_size=50, transform=basic_transform, path='./MNIST_data'):
    train_data = datasets.MNIST(path, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(path, train=False, download=True, transform=transform)
    train_loader = utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def svhn(batch_size=50, transform=as_MNIST_transform, path='./SVHN_data'):
    train_data = datasets.SVHN(path, split='train', download=True, transform=transform)
    test_data = datasets.SVHN(path, split='test', download=True, transform=transform)
    train_loader = utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_loader(first_set, second_set, device='cpu', merge=True):
    second_iter = iter(second_set)
    for batch_idx, (first_data, first_target) in enumerate(first_set):
        try:
            second_data, second_target = next(second_iter)
        except StopIteration:
            second_iter = iter(second_set)
            second_data, second_target = next(second_iter)
        ones_target = torch.ones(size=(first_data.shape[0], 1), dtype=torch.float32, device=device)
        zeros_target = torch.zeros(size=(second_data.shape[0], 1), dtype=torch.float32, device=device)
        
        if merge:
            data = torch.cat([first_data, second_data], 0).view(-1, 784).to(device)
            target = torch.cat([first_target, second_target], 0).to(device)
            domain = torch.cat([ones_target, zeros_target], 0)
            yield batch_idx, data, target, domain
        else:
            yield batch_idx, (first_data.to(device), first_target.to(device)),\
                  (second_data.to(device), second_target.to(device)), (ones_target, zeros_target)
        
def plot_imgs(images, shape):
    fig = plt.figure(figsize=shape[::-1], dpi=80)
    for j in range(1, len(images) + 1):
        ax = fig.add_subplot(shape[0], shape[1], j)
        ax.matshow(images[j - 1, :, :, :], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()

def plot_results(model, loader, shape):
    data, target = next(iter(loader))
    with torch.no_grad():
        output = model(data)
    pred = output.data.max(1, keepdim=True)[1]
    plot_mnist(data.data.numpy(), shape)
    print(pred.numpy().reshape(shape))
        
def plot_graphs(log, tpe='loss'):
    keys = log.keys()
    logs = {k:[z for z in zip(*log[k])] for k in log.keys()}
    epochs = {k:range(len(log[k])) for k in log.keys()}
    
    if tpe == 'loss':
        handlers, = zip(*[plt.plot(epochs[k], logs[k][0], label=k) for k in log.keys()])
        plt.title('errors')
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.legend(handles=handlers)
        plt.show()
    elif tpe == 'accuracy':
        handlers, = zip(*[plt.plot(epochs[k], logs[k][1], label=k) for k in log.keys()])
        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(handles=handlers)
        plt.show()
    
def to_onehot(x, n, device='cpu'):
    one_hot = torch.FloatTensor(x.size(0), n, device=device).zero_()
    one_hot.scatter_(1, x[:, None], 1)
    return one_hot

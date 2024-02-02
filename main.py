# Library Imports
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# Local Imports
import dataloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # For CIFAR10
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        # For MNIST
        #self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Program Arguments ================================================================================================
    parser = argparse.ArgumentParser(description="Deep Learning Optimizers Testing")
    # Program
    parser.add_argument('--threads', default=1, type=int, help='number of threads')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='Random seed (default: 1)')

    # Dataset options
    parser.add_argument('--dataset', default='cifar10', help='cifar10')
    parser.add_argument('--datapath', default='cifar10/data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='no data preprocessing')
    parser.add_argument('--data_split', default=1, type=int,
                        help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int,
                        help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    # Training options
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='Input batch size for testing (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='Input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='Number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='Learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='How many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Loading Datasets =================================================================================================
    # CIFAR10
    datasets.CIFAR10(root=args.dataset + '/data', train=True, download=True)
    train_loader, test_loader = dataloader.load_dataset(args.dataset, args.datapath, args.batch_size, args.threads,
                                                        args.raw_data, args.data_split, args.split_idx,
                                                        args.trainloader, args.testloader)

    # MNIST
    #transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.1307,), (0.3081,))
    #])
    #train_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    #test_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Initialize and train model =======================================================================================
    model = Net().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)  # TODO: allow multiple different optimizers
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()

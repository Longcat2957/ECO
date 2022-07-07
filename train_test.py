import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision
from utils.transform import VideoTransform
from utils.dataset import UCF101Dataset

traindataset = UCF101Dataset(
    root='./data/UCF-101',
    annotation_path='./data/UCF_annotations',
    frames_per_clip=16,
    train=True,
    step_between_clips=4,
    num_workers=8,
    transform=VideoTransform(resize=(224, 224), phase='train')
    )


valdataset = UCF101Dataset(
    root='./data/UCF-101',
    annotation_path='./data/UCF_annotations',
    frames_per_clip=16,
    step_between_clips=4,
    train=False,
    num_workers=8,
    transform=VideoTransform(resize=(224, 224), phase='val')
)

train_dataloader = DataLoader(traindataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(valdataset, batch_size=1, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device} device')

from model.eco_lite import ECO_Lite
mymodel = ECO_Lite().to(device)

epochs = 5
learning_rate = 1e-5

my_loss_fn = nn.CrossEntropyLoss()
my_optimizer = torch.optim.SGD(mymodel.parameters(),lr=learning_rate, momentum=0.9, dampening=0, nesterov=True)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, mymodel, my_loss_fn, my_optimizer)
    test_loop(val_dataloader, mymodel, my_loss_fn)
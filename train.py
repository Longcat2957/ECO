import torch
from tqdm import tqdm

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs, train_intervals):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device = {device}')
    net.to(device)
    print(f'load model to {device}')

    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        print('EPOCH {}/{}'.format(epoch+1, num_epochs))
        print('------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            
            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == 'train'):
                continue
            
            for inputs, _, labels, _ in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    
        if epoch % train_intervals == 0 and epoch != 0:
            torch.save(net, './weights/{}.pth'.format(epoch))
    
    torch.save(net, './weights/{}_final.pth'.format(num_epochs))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import mnist

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet18(weights=None, num_classes=10).to(device)
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    state_dict = torch.load('./moco.pth')
    for k in list(state_dict.keys()):
        if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
            state_dict[k[len("encoder_q.") :]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    aug = transforms.Compose([
        transforms.Resize(224, 3),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = mnist.MNIST('./data', train=True, download=True, transform=aug)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataset = mnist.MNIST('./data', train=False, download=True, transform=aug)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    epoch = 50
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epoch)

    for e in range(epoch):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            logits = model(x)

            loss = loss_func(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 100 == 0:
                print(f'epoch: [{e+1}/{epoch}] step: [{step}/{len(train_loader)}] loss: {loss.item():.4f}')

        cosine_scheduler.step()

        model.eval()
        total_acc = 0
        total_num = 0

        for step, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                logits = model(x)

            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)
            total_acc += torch.eq(pred, y).sum().item()
            total_num += len(x)

        print(f'epoch: [{e+1}/{epoch}] acc: {total_acc / total_num:.4f}')
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import mnist
from torchvision import transforms

from moco import MoCo


class MnistDataset(Dataset):

    def __init__(self, root, train=True, download=True, aug=None):
        self.dataset = mnist.MNIST(root=root, train=train, download=download)
        self.aug = aug

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        q, _ = self.dataset[i]
        k, _ = self.dataset[i]
        if self.aug:
            q = self.aug(q)
            k = self.aug(k)
        return q, k


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    aug = transforms.Compose([
        transforms.Resize(224, 3),
        transforms.Grayscale(num_output_channels=3),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = MnistDataset('./data', train=True, aug=aug)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

    moco = MoCo(dim=128, K=65536, m=0.999, T=0.07).to(device)

    epoch = 20
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(moco.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epoch)

    moco.train()
    for e in range(epoch):
        for step, (q, k) in enumerate(train_loader):
            q, k = q.to(device), k.to(device)

            logits, labels = moco(q, k)

            loss = loss_func(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 100 == 0:
                print(f"epoch: [{e+1}/{epoch}] step: [{step+1}/{len(train_loader)}] loss: {loss.item():.4f}")

        cosine_scheduler.step()

        print("========================================")

    torch.save(moco.state_dict(), 'moco.pth')

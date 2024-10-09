import torch
import torch.nn as nn
from torchvision import models


class MoCo(nn.Module):

    def __init__(self, dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = models.resnet18(weights=None, num_classes=dim)
        self.encoder_k = models.resnet18(weights=None, num_classes=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data) # 初始化参数
            param_k.requires_grad = False # encoder_k不需要更新

        # 创建队列
        self.register_buffer('queue', torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # 动量更新 k(t) = m * k(t-1) + (1 - m) * q(t)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = self.m * param_k.data + (1.0 - self.m) * param_q.data

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        b = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % b == 0 # K一定要被b整除

        self.queue.data[:, ptr: ptr+b] = keys.T
        ptr = (ptr + b) % self.K # 移动指针

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):

        # 计算query
        q = self.encoder_q(im_q) # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # 计算key
        with torch.no_grad():
            self._momentum_update_key_encoder() # 更新key encoder

            # 因为这儿是个demo单卡训练, 所以不需要源码中对多卡的BN操作
            k = self.encoder_k(im_k) # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # 计算logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # 队列更新
        self._dequeue_and_enqueue(k)

        return logits, labels


if __name__ == '__main__':
    base_encoder = models.resnet18(num_classes=128)

    moco = MoCo(
        base_encoder=base_encoder,
    )

    im_q = torch.randn(2, 3, 224, 224)
    im_k = torch.randn(2, 3, 224, 224)

    moco.forward(im_q, im_k)
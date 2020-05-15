import time
from lcfcn import lcfcn_loss
import torch



if __name__ == "__main__":
    n, c, h, w = 1, 3, 100, 100
    points = torch.cuda.FloatTensor(n, h, w).uniform_() > 0.8
    logits = torch.randn(n, c,h,w)

    s_time = 0.
    for i in range(10):
        # compute loss given 'points' as HxW mask (1 pixel label per object)
        loss = lcfcn_loss.compute_lcfcn_loss(logits, points)
        print(loss)
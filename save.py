import time, os
import numpy as np
import torch

def save(model, epoch, path):
    if not os.path.exists(path):
        os.makedirs(path)
    model_out_path = "{}/model_epoch_{}.pth".format(path, epoch)
    state = {"epoch": epoch, "model": model}

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

def save_G1D2(G, D1, D2, epoch, path):
    if not os.path.exists(path):
        os.makedirs(path)
    model_out_path = "{}/model_epoch_{}.pth".format(path, epoch)
    state = {"epoch": epoch, "G": G, "D1": D1, "D2": D2}

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


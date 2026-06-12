import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
from skimage import data
import numpy as np
from scipy.fft import dst, idst, dct, idct
from scipy.optimize import brentq
from skimage.transform import resize
import torch
import torch.nn as nn
from skimage import data as skdatas
from PIL import Image
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
from scipy import ndimage
import json
from IPython.display import Audio

def train(model, n_idx_gpu, v_tensor_gpu, steps1=1000, steps2=0, lr=1e-3):
    psnr_curve = []
    total_steps = steps1 + steps2

    with tqdm(total=total_steps, desc="Training", unit="step") as pbar:
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = torch.optim.Adam([
            {"params": model.W1.parameters(), "lr": 5e-2},
            {"params": model.W2.parameters(), "lr": 7e-1},
            {"params": model.W3.parameters(), "lr": 3e-1},
        ])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode = 'max',
            factor = 0.5,
            patience = 50
        )
        best_psnr, best_state = 0, None
        
        for step in range(steps1):
            loss = torch.mean((model(n_idx_gpu) - v_tensor_gpu)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            psnr = 10 * np.log10(1.0 / loss.item())
            scheduler.step(psnr)
            psnr_curve.append(psnr)
            if psnr > best_psnr:
                best_psnr = psnr
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            # pbar.set_postfix(lr=f"{lr}", psnr=f"{psnr:.2f}", best_psnr=f"{best_psnr:.2f}")
            pbar.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}", psnr=f"{psnr:.2f}", best_psnr=f"{best_psnr:.2f}")
            pbar.update(1)

    return best_psnr, best_state, psnr_curve

        # # Phase 1: constant lr
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # for step in range(steps1):
        #     loss = torch.mean((model(n_idx_gpu) - v_tensor_gpu)**2)
        #     optimizer.zero_grad(); loss.backward(); optimizer.step()
        #     psnr = 10 * np.log10(1.0 / loss.item())
        #     psnr_curve.append(psnr)
        #     pbar.set_postfix(phase=1, psnr=f"{psnr:.2f}")
        #     pbar.update(1)

        # # Phase 2: reduce on plateau
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='max', factor=0.5, patience=50)
        # best_psnr, best_state = 0, None
        # for step in range(steps2):
        #     loss = torch.mean((model(n_idx_gpu) - v_tensor_gpu)**2)
        #     optimizer.zero_grad(); loss.backward(); optimizer.step()
        #     psnr = 10 * np.log10(1.0 / loss.item())
        #     scheduler.step(psnr)
        #     psnr_curve.append(psnr)
        #     if psnr > best_psnr:
        #         best_psnr = psnr
        #         best_state = {k: v.clone() for k, v in model.state_dict().items()}
        #     pbar.set_postfix(phase=2, psnr=f"{psnr:.2f}", best_psnr=f"{best_psnr:.2f}")
        #     pbar.update(1)

class TwoLayerSIREN(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.W1 = nn.Linear(1, M, bias=True)
        self.W2 = nn.Linear(M, M, bias=False)
        self.W3 = nn.Linear(M, 1, bias=False)

    def forward(self, x):
        h = torch.sin(self.W1(x))
        h = torch.sin(self.W2(h))
        return self.W3(h)

# Bessel inversion — match |amp| then restore sign
def find_a(amp):
    try:
        return brentq(lambda a: 2*jv(1, a) - abs(amp), 1e-6, 1)
    except ValueError:
        return abs(amp)/2
    
def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b)**2))
    
def JA_SIREN(data, M):
    N = len(data)
    V = dst(data, type=2)
    # print(len(V))
    vector = [1]+[0]*99
    vector = np.tile(vector, len(V)//len(vector)+1)  # 30000 * 3 = 90,000
    # print(len(vector))
    vector = vector[:len(V)]
    # print(len(vector))
    V = V*vector
    magnitudes = np.abs(V)
    sorted_idx = np.argsort(magnitudes)[::-1]

    ## Compute compositional parameters
    primary_idx = sorted_idx[:M]
    primary_amps = V[primary_idx] / N  # signed
    a_comps = np.array([find_a(a) for a in primary_amps])
    signs   = np.sign(primary_amps)

    ## Build model
    model  = TwoLayerSIREN(M)
    w1 = np.pi * (primary_idx + 1) / 2
    b1 = np.pi * (primary_idx + 1) * (1 + 1/N) / 2
    w2 = np.diag(a_comps)
    w3 = signs

    with torch.no_grad():
        model.W1.weight.copy_(torch.tensor(w1, dtype=torch.float32).unsqueeze(1))
        model.W1.bias.copy_(torch.tensor(b1, dtype=torch.float32))
        model.W2.weight.copy_(torch.tensor(w2, dtype=torch.float32))
        model.W3.weight.copy_(torch.tensor(w3, dtype=torch.float32).unsqueeze(0))

    return model
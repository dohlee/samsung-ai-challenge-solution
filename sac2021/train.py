import wandb

import argparse
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sac2021.sac2021 import AtomTransformer
from sac2021.data import SACData
from sklearn.model_selection import KFold

import sac2021.const as const
import sac2021.loss as loss
from sac2021.scheduler import GradualWarmupScheduler

torch.autograd.set_detect_anomaly(True)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', required=True)
    # parser.add_argument('--ckpt', required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--loss', required=True)
    parser.add_argument('--bsz', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=24)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=16)
    parser.add_argument('--d-ff', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--res-p', type=float, default=0.0)
    parser.add_argument('--att-p', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--augs', nargs='+', default=[])
    parser.add_argument('--debug', action='store_true', default=False)

    return parser.parse_args()

args = parse_arguments()

random.seed(args.seed)
np.random.seed(args.seed)
os.environ["PYTHONHASHSEED"] = str(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)  # type: ignore
# torch.backends.cudnn.deterministic = True  # type: ignore
# torch.backends.cudnn.benchmark = True  # type: ignore

if args.debug:
    os.environ['WANDB_MODE'] = 'offline'

meta = pd.read_csv(const.fp['train_meta'])
meta = meta[~meta.uid.isin(const.ignored_uids)].reset_index(drop=True)

td_meta = meta[(meta.uid.str.startswith('train')) | (meta.uid.str.startswith('dev'))].reset_index(drop=True)

losses = {
    'mse': nn.MSELoss,
    'mae': nn.L1Loss,
    'logcosh': loss.LogCoshLoss,
}
criterion = losses[args.loss]()
mae = nn.L1Loss()

# W&B configuration.
wandb.init(project='sac', entity='dohlee')
config = wandb.config
config.update(args)
config.update({'pretrained': False})
# config.update({'pretrained_model': args.ckpt})
config.update({'data_version': SACData.version})

cv = KFold(n_splits=40, shuffle=True, random_state=args.seed)

for fold, (train_idx, val_idx) in enumerate(cv.split(td_meta), 1):
    if fold == args.fold:
        break

val_idx_set = set(val_idx)
train_idx = [idx for idx in range(len(meta)) if idx not in val_idx_set]

print('train', len(train_idx))
print('val', len(val_idx))

net = AtomTransformer(config.n_layers, config.n_heads, config.d_model, config.d_ff, res_p=args.res_p, att_p=args.att_p)
# Load checkpoint.
# ckpt = torch.load(args.ckpt)
# net.load_state_dict(ckpt['net'], strict=False)
# Freeze parameters. Will be unfreezed later.
# for param in net.atom_embedding.parameters():
    # param.requires_grad = False
# for param in net.transformer.parameters():
    # param.requires_grad = False

# for i in [-1, -2, -3]:
    # for param in net.transformer.layers[i].parameters():
        # param.requires_grad = True
net.cuda()

wandb.watch(net, log='all')
optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, 
    patience=15, threshold=0.005, threshold_mode='rel'
)

train_dataset = SACData(idx=train_idx, augs=args.augs)
val_dataset = SACData(idx=val_idx)
train_loader = DataLoader(
    train_dataset, num_workers=16, batch_size=config.bsz, shuffle=True, pin_memory=True, drop_last=True,
)
val_loader = DataLoader(
    val_dataset, num_workers=16, batch_size=config.bsz, shuffle=False, pin_memory=True, drop_last=False,
)
n_batch = len(train_loader)

optimizer.zero_grad()
optimizer.step()

best_val_mae, val_mae = 10000, 10000
for epoch in range(1, 501):
    scheduler.step(val_mae)
    # scheduler.step(epoch=epoch, metrics=val_mae)
    # print(f'LR={get_lr(optimizer)}')
    # print(scheduler.get_lr())

    # losses
    running_gap_loss = 0.0
    running_s1_loss = 0.0
    running_t1_loss = 0.0

    # metrics (MAE)
    running_gap_mae = 0.0
    running_gap_mae_secondary = 0.0
    running_gap_mae_tertiary = 0.0

    net.train()
    for batch, data in enumerate(train_loader, 1):
        for k, v in data.items():
            data[k] = v.cuda()

        optimizer.zero_grad()

        atom_idx = data['atom_idx']
        hyb = data['hyb']
        donac = data['donac']
        spin = data['spin']
        feat = data['feat']
        pdist = data['pdist']
        sym = data['sym']
        angle = data['angle']
        adj = data['adj']
        mask = data['mask']
        fp = data['fp']
        out_mask = data['out_mask']
        n_atoms = data['n_atoms']
        gap_target = data['target'].float()
        s1_target = data['s1'].float()
        t1_target = data['t1'].float()

        gap_out, s1_out, t1_out = net(atom_idx, hyb, donac, spin, feat, pdist, angle, adj, mask, out_mask, n_atoms)
        _bsz = gap_out.size(0)
        gap_loss = criterion(gap_out, gap_target.view(_bsz, 1))
        s1_loss = criterion(s1_out, s1_target.view(_bsz, 1))
        t1_loss = criterion(t1_out, t1_target.view(_bsz, 1))
        loss = gap_loss + 0.05 * (s1_loss + t1_loss)
        # loss = s1_loss + t1_loss

        running_gap_loss += gap_loss.detach().item()
        running_s1_loss += s1_loss.detach().item()
        running_t1_loss += t1_loss.detach().item()

        _mae = mae(gap_out, gap_target.view(_bsz, 1))
        running_gap_mae += _mae.detach().item()

        _mae = mae(s1_out - t1_out, gap_target.view(_bsz, 1))
        running_gap_mae_secondary += _mae.detach().item()

        _mae = mae(0.5 * gap_out + 0.5 * (s1_out - t1_out) , gap_target.view(_bsz, 1))
        running_gap_mae_tertiary += _mae.detach().item()

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        if batch % 10 == 0:
            batch_gap_loss = running_gap_loss / 10.
            batch_s1_loss = running_s1_loss / 10.
            batch_t1_loss = running_t1_loss / 10.
            batch_gap_mae = running_gap_mae / 10.
            batch_gap_mae_secondary = running_gap_mae_secondary / 10.
            batch_gap_mae_tertiary = running_gap_mae_tertiary / 10.

            print(f'E{epoch:<3}B{batch:<3} (lr={get_lr(optimizer):.3g})l={batch_gap_loss:.4f} l2={batch_s1_loss:.4f} l3={batch_t1_loss:.4f} mae={batch_gap_mae:.4f} mae2={batch_gap_mae_secondary:.4f} mae3={batch_gap_mae_tertiary:.4f} lmae={np.log(batch_gap_mae):.4f}')

            running_gap_loss = 0.0
            running_s1_loss = 0.0
            running_t1_loss = 0.0
            running_gap_mae = 0.0
            running_gap_mae_secondary = 0.0
            running_gap_mae_tertiary = 0.0
            
            log_dict = {
                'train/loss': batch_gap_loss,
                'train/s1_loss': batch_s1_loss,
                'train/t1_loss': batch_t1_loss,
                'train/mae': batch_gap_mae, 
                'train/mae_secondary': batch_gap_mae_secondary,
                'train/mae_tertiary': batch_gap_mae_tertiary,
                'train/lr': get_lr(optimizer)
            }
            wandb.log(log_dict)

    net.eval()
    val_gap_outs, val_s1_outs, val_t1_outs = [], [], []
    val_gap_targets, val_s1_targets, val_t1_targets = [], [], []
    with torch.no_grad():
        for batch, data in enumerate(val_loader, 1):
            for k, v in data.items():
                data[k] = v.cuda()

            atom_idx = data['atom_idx']
            hyb = data['hyb']
            donac = data['donac']
            spin = data['spin']
            feat = data['feat']
            pdist = data['pdist']
            sym = data['sym']
            angle = data['angle']
            adj = data['adj']
            mask = data['mask']
            fp = data['fp']
            out_mask = data['out_mask']
            n_atoms = data['n_atoms']
            target = data['target'].float()
            s1 = data['s1'].float()
            t1 = data['t1'].float()

            gap_out, s1_out, t1_out = net(atom_idx, hyb, donac, spin, feat, pdist, angle, adj, mask, out_mask, n_atoms)

            _bsz = gap_out.size(0)
            val_gap_outs.append(gap_out.cpu())
            val_s1_outs.append(s1_out.cpu())    
            val_t1_outs.append(t1_out.cpu())
            val_gap_targets.append(target.cpu().view(_bsz, -1))
            val_s1_targets.append(s1.cpu().view(_bsz, -1))
            val_t1_targets.append(t1.cpu().view(_bsz, -1))

    val_gap_outs, val_s1_outs, val_t1_outs = map(lambda x: torch.cat(x, dim=0), (val_gap_outs, val_s1_outs, val_t1_outs))
    val_gap_targets, val_s1_targets, val_t1_targets = map(lambda x: torch.cat(x, dim=0), (val_gap_targets, val_s1_targets, val_t1_targets))

    val_gap_loss = criterion(val_gap_outs, val_gap_targets)
    val_s1_loss = criterion(val_s1_outs, val_s1_targets)
    val_t1_loss = criterion(val_t1_outs, val_t1_targets)

    val_mae = mae(val_gap_outs, val_gap_targets)
    val_mae_secondary = mae(val_s1_outs - val_t1_outs, val_gap_targets)
    val_mae_tertiary = mae(0.5 * val_gap_outs + 0.5 * (val_s1_outs - val_t1_outs), val_gap_targets)

    print(f'***Validation E{epoch:<3} loss={val_gap_loss:.4f} s1_loss={val_s1_loss:.4f} t1_loss={val_t1_loss:.4f} mae={val_mae:.4f} mae2={val_mae_secondary:.4f} mae3={val_mae_tertiary:.4f} lmae={np.log(val_mae):.4f}')
    log_dict = {
        'val/loss': val_gap_loss,
        'val/mae': val_mae,
        'val/mae_secondary': val_mae_secondary,
        'val/mae_tertiary': val_mae_tertiary,
        'epoch': epoch,
    }
    wandb.log(log_dict)

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        wandb.summary['best_val_mae'] = best_val_mae

        ckpt = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_mae': best_val_mae,
        }
        torch.save(ckpt, args.output)

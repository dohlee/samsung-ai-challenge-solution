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
from sac2021.sac2021 import AtomTransformerPretrain as AtomTransformer
from sac2021.data import SACData
from sklearn.model_selection import KFold

import sac2021.const as const
import sac2021.loss as loss
from scheduler import GradualWarmupScheduler

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
    parser.add_argument('--res-p', type=float, default=0.0)
    parser.add_argument('--att-p', type=float, default=0.0)
    parser.add_argument('--d-ff', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-5)
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

meta = pd.read_csv(const.fp['pretrain_meta'])
meta = meta[~meta.uid.isin(const.ignored_uids)].reset_index(drop=True)

# td_meta = meta[(meta.uid.str.startswith('train')) | (meta.uid.str.startswith('dev'))].reset_index(drop=True)

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

for fold, (train_idx, val_idx) in enumerate(cv.split(meta), 1):
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
    patience=5, threshold=0.005, threshold_mode='abs'
)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
    # optimizer, milestones=[20, 30, 40, 50, 60], gamma=0.5,
# )
# scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler)

train_dataset = SACData(idx=train_idx, augs=args.augs, pretrain=True)
val_dataset = SACData(idx=val_idx, pretrain=True)
train_loader = DataLoader(
    train_dataset, num_workers=16, batch_size=config.bsz, shuffle=True, pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, num_workers=16, batch_size=config.bsz, shuffle=False, pin_memory=True, drop_last=False,
)
n_batch = len(train_loader)

optimizer.zero_grad()
optimizer.step()

best_val_mae, val_mae = 10000, 10000
for epoch in range(1, 71):
    scheduler.step(val_mae)
    # scheduler.step(epoch=epoch, metrics=val_mae)
    # print(f'LR={get_lr(optimizer)}')
    # print(scheduler.get_lr())

    # losses
    running_homo_loss = 0.0
    running_lumo_loss = 0.0

    # metrics (MAE)
    running_homo_mae = 0.0
    running_lumo_mae = 0.0

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
        angle = data['angle']
        adj = data['adj']
        mask = data['mask']
        out_mask = data['out_mask']
        n_atoms = data['n_atoms']
        homo_target = data['homo'].float()
        lumo_target = data['lumo'].float()

        homo_out, lumo_out = net(atom_idx, hyb, donac, spin, feat, pdist, angle, adj, mask, out_mask, n_atoms)
        _bsz = homo_out.size(0)
        homo_loss = criterion(homo_out, homo_target.view(_bsz, 1))
        lumo_loss = criterion(lumo_out, lumo_target.view(_bsz, 1))

        loss = 0.5 * (homo_loss + lumo_loss)

        running_homo_loss += homo_loss.detach().item()
        running_lumo_loss += lumo_loss.detach().item()

        _mae = mae(homo_out, homo_target.view(_bsz, 1))
        running_homo_mae += _mae.detach().item()

        _mae = mae(lumo_out, lumo_target.view(_bsz, 1))
        running_lumo_mae += _mae.detach().item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1) # gradient clipping.
        optimizer.step()

        if batch % 10 == 0:
            batch_homo_loss = running_homo_loss / 10.
            batch_lumo_loss = running_lumo_loss / 10.

            batch_homo_mae = running_homo_mae / 10.
            batch_lumo_mae = running_lumo_mae / 10.
            batch_homo_lmae = np.log(batch_homo_mae)
            batch_lumo_lmae = np.log(batch_lumo_mae)

            print(f'E{epoch:<3}B{batch:<3} hl={batch_homo_loss:.4f} ll={batch_lumo_loss:.4f} hmae={batch_homo_mae:.4f} lmae={batch_lumo_mae:.4f} lhmae={batch_homo_lmae:.4f} llmae={batch_lumo_lmae:.4f}')

            running_homo_loss = 0.0
            running_lumo_loss = 0.0
            running_homo_mae = 0.0
            running_lumo_mae = 0.0
            
            log_dict = {
                'train/homo_loss': batch_homo_loss,
                'train/lumo_loss': batch_lumo_loss,
                'train/homo_mae': batch_homo_mae, 
                'train/lumo_mae': batch_lumo_mae,
                'train/lr': get_lr(optimizer)
            }
            wandb.log(log_dict)

    net.eval()
    val_homo_outs, val_lumo_outs = [], []
    val_homo_targets, val_lumo_targets = [], []
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
            angle = data['angle']
            adj = data['adj']
            mask = data['mask']
            out_mask = data['out_mask']
            n_atoms = data['n_atoms']
            homo = data['homo'].float()
            lumo = data['lumo'].float()

            homo_out, lumo_out = net(atom_idx, hyb, donac, spin, feat, pdist, angle, adj, mask, out_mask, n_atoms)

            _bsz = homo_out.size(0)
            val_homo_outs.append(homo_out.cpu())
            val_lumo_outs.append(lumo_out.cpu())
            val_homo_targets.append(homo.cpu().view(_bsz, -1))
            val_lumo_targets.append(lumo.cpu().view(_bsz, -1))

    val_homo_outs, val_lumo_outs = map(lambda x: torch.cat(x, dim=0), (val_homo_outs, val_lumo_outs))
    val_homo_targets, val_lumo_targets = map(lambda x: torch.cat(x, dim=0), (val_homo_targets, val_lumo_targets))

    homo_loss = criterion(val_homo_outs, val_homo_targets)
    lumo_loss = criterion(val_lumo_outs, val_lumo_targets)

    homo_mae = mae(val_homo_outs, val_homo_targets)
    lumo_mae = mae(val_lumo_outs, val_lumo_targets)
    homo_lmae = torch.log(homo_mae)
    lumo_lmae = torch.log(lumo_mae)

    print(f'***Validation E{epoch:<3} hl={homo_loss:.4f} ll={lumo_loss:.4f} hmae={homo_mae:.4f} lmae={lumo_mae:4f} hlmae={homo_lmae:.4f} llmae={lumo_lmae:.4f}')
    log_dict = {
        'val/homo_loss': homo_loss,
        'val/lumo_loss': lumo_loss,
        'val/homo_mae': homo_mae,
        'val/lumo_mae': lumo_mae,
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

    if epoch % 3 == 0:
        ckpt = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_mae': best_val_mae,
        }
        torch.save(ckpt, args.output +f'.e{epoch}')


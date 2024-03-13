import numpy as np
import random,json
import torch
import torch.nn as nn
import torch.optim
from transformer.model import*
from utils.dataset import *
from utils.preprocessing import *
from utils.trainer import *

import warmup_scheduler
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from colorama import Fore, Style
import os
import torch.optim.lr_scheduler as lr_schedule
import argparse
from tqdm import trange

parser = argparse.ArgumentParser(description='Image Captioning on Flickr8k quick training script')

# Data args
parser.add_argument('--data_path', default='./data', type=str, help='dataset path')
parser.add_argument('--max_seq_len', default=60, type=int, help='max sequence length')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--freq', default=10, type=int, metavar='N', help='log frequency (by iteration)')

# Model parameters
parser.add_argument('--height', default=224, type=int, metavar='N', help='image height')
parser.add_argument('--width', default=224, type=int, metavar='N', help='image width')
parser.add_argument('--channel', default=3, type=int, help='disable cuda')
parser.add_argument('--enc_heads', default=8, type=int, help='number of encoder  heads')
parser.add_argument('--enc_depth', default=4, type=int, help='number of encoder blocks')
parser.add_argument('--dec_heads', default=8, type=int, help='number of decoder  heads')
parser.add_argument('--dec_depth', default=1, type=int, help='number of decoder blocks')
parser.add_argument('--patch_size', default=16, type=int, help='patch size')
parser.add_argument('--dim', default=512, type=int, help='embedding dim of patch')
parser.add_argument('--enc_mlp_dim', default=1024, type=int, help='feed forward hidden_dim for an encoder block')
parser.add_argument('--dec_mlp_dim', default=1024, type=int, help='feed forward hidden_dim for a decoder block')


# Optimization hyperparams
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--warmup', default=5, type=int, metavar='N', help='number of warmup epochs')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
parser.add_argument('--lr', default=0.0003, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default=False, help='Version')

args = parser.parse_args()
data_location = args.data_path
lr = args.lr
weight_decay = args.weight_decay
height, width, n_channels = args.height, args.width, args.channel
patch_size, dim, enc_head = args.patch_size, args.dim, args.enc_heads
enc_feed_forward, enc_depth = args.enc_mlp_dim, args.enc_depth
dec_feed_forward, dec_depth = args.dec_mlp_dim, args.dec_depth
dec_head = args.dec_heads

batch_size = args.batch_size
warmup = args.warmup


device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


if __name__ == '__main__':
    
    transforms =  T.Compose([
                        T.Resize((height,width)),
                        T.ToTensor()
                         ])

    data = FlickrDataset(
    root_dir = data_location+"/Images",
    captions_file = data_location+"/captions.txt",
    transform = transforms
    )
    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, test_size])
    train_loader = DataLoader(
                    dataset=train_data,
                    batch_size=batch_size,
                    num_workers=args.j,
                    shuffle=True,
                    collate_fn = collate_fn
                    )
    val_loader = DataLoader(
                    dataset=val_data,
                    batch_size=batch_size,
                    num_workers=args.j,
                    shuffle=False,
                    collate_fn = collate_fn
                    )

    vocab_size = len(data.vocab)
    max_seq_len = args.max_seq_len
    padding_idx = data.vocab.stoi['<PAD>']
    with open('vocab.json', 'w') as f:
        # use json.dump to write the dictionary to the file
        json.dump(data.vocab.stoi, f)
    model = Transformer(height,width,n_channels,patch_size,dim,enc_head,enc_feed_forward,enc_depth,
                    dec_head,dec_feed_forward,dec_depth,max_seq_len,vocab_size,padding_idx)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay = weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx).to(device)
    num_epochs = args.epochs
    base_scheduler = lr_schedule.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=warmup, after_scheduler=base_scheduler)

    # Train the model
    best_loss = float('inf')
    torch.autograd.set_detect_anomaly(True)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        final_epoch = args.epochs
        num_epochs = final_epoch - (checkpoint['epoch'] + 1)

    print(Fore.LIGHTGREEN_EX+'='*100)
    print("[INFO] Begin training for {0} epochs".format(num_epochs))
    print('='*100+Style.RESET_ALL)

    for epoch in trange(num_epochs,ncols=100):
        train_loss = train_epoch(model,train_loader,optimizer,criterion,device)
        valid_loss = validate(model,val_loader,criterion,device)
        scheduler.step()
        torch.cuda.empty_cache()
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {valid_loss}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, f"captioning.pt")

    print(Fore.GREEN+'='*100)
    print("[INFO] End training")
    print('='*100+Style.RESET_ALL)
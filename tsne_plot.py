import os, time, gc, json, pickle, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn import DataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, Conv1D
from tensorboardX import SummaryWriter
from tqdm import tqdm
import importlib
import logging
import copy

from data.util import *
from util import *
from model import VAEModel

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

devices = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = devices

# specify for the trained VAE model
add_input = True
add_softmax = False
add_attn = False

parser = argparse.ArgumentParser()

# global parameters
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=16, type=int)

# use GPU
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--no_gpu', action="store_true")

parser.add_argument('--model_type', type=str, default='t0', choices=['t0', 't1'], help="t: type")
parser.add_argument('--dataset', type=str, default='yp', choices=['ax', 'yp', 'wp', 'wi'],
                    help="Dataset to use for training")
parser.add_argument('--load', type=str, default='out/yelp.2/', help='path to load model from')

parser.add_argument('--data-dir', type=str, default='data')
parser.add_argument('--out-dir', type=str, default='out')

if sys.argv[1:] == ['--mode=server']:
    args = parser.parse_args([])   # run in pycharm console
else:
    args = parser.parse_args()  # run in cmd

# gpu
if not torch.cuda.is_available(): args.no_gpu = True
gpu = not args.no_gpu
if gpu: torch.cuda.set_device(args.gpu)
device = torch.device(args.gpu if gpu else "cpu")

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if gpu: torch.cuda.manual_seed(args.seed)

# logging
save_folder = os.path.join(args.load)
os.makedirs(save_folder, exist_ok=True)
logging.basicConfig(filename=os.path.join(save_folder, 'tSNE.log'),
                    level=logging.INFO, format='%(asctime)s--- %(message)s')

print('Loading models...')
cache_dir = os.path.join(args.out_dir, 'model_cache')
os.makedirs(cache_dir, exist_ok=True)
# Load pre-trained teacher tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
# Hack to allow tokenizing longer sequences.
tokenizer.max_len = int(1e12)
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir)
print('gpt2_params:', num_params(gpt2_model))  # gpt2: 124439808
config = GPT2Config()

# add special tokens
special_tokens_dict = {
    'pad_token': '<|startoftext|>',
    'cls_token': '<|startofcond|>',
    'sep_token': '<|sepofcond|>',
    'mask_token': '<|endofcond|>'
}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print('We have added', num_added_toks, 'special tokens')
# Notice: resize_token_embeddings expect to receive the full size of the new vocab
gpt2_model.resize_token_embeddings(len(tokenizer))
assert tokenizer.pad_token == '<|startoftext|>'

VAE = VAEModel(config, add_input=add_input, add_attn=add_attn, add_softmax=add_softmax)
init_para_frompretrained(VAE.transformer, gpt2_model.transformer, share_para=True)
init_para_frompretrained(VAE.encoder, gpt2_model.transformer, share_para=False)
VAE.lm_head.weight = gpt2_model.lm_head.weight
if VAE.add_softmax:
    VAE.lm_head_rep = Conv1D(*gpt2_model.lm_head.weight.size())
print('VAE_params:', num_params(VAE))  # 286694400
print('Done.')

print('Loading model weights...')
state = torch.load(os.path.join(args.load, 'model_latest.pt'), map_location='cpu')
if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
    state_copy = copy.copy(state)
    keys = state_copy.keys()
    for k in keys:
        state[k.replace('module.', '')] = state.pop(k)
VAE.load_state_dict(state)
VAE.eval()
VAE = VAE.to(device)
print('Done.')

print('Setup data...')
test_loader = prepare_dataset(
    args.data_dir, args.dataset, tokenizer,
    1, 1024, 1, 1024, args.batch_size, 1024,
    make_train=False, make_val=False, make_test=True, model_type=args.model_type
)[0]
print('Done.')

# get embedding
X_emb = None
y = None

# test_iter = iter(test_loader); c_mask, c_tokens, x_mask, x_tokens, input_tokens, target_tokens, mask = next(test_iter)
with torch.no_grad():
    with tqdm(total=len(test_loader)) as pbar:
        for i, (c_mask, c_tokens, x_mask, x_tokens, input_tokens, target_tokens, mask) in enumerate(test_loader):
            x_mask = x_mask.to(device)
            x_tokens = x_tokens.to(device)
            latent_mean, _ = VAE.encoder(input_ids=x_tokens, attention_mask=x_mask)[:2]

            if i == 0:
                X_emb = latent_mean.data
                y = [tokenizer.decode(l)[:2] for l in c_tokens.tolist()]
            else:
                X_emb = torch.cat((X_emb, latent_mean.data), dim=0)
                y.extend([tokenizer.decode(l)[:2] for l in c_tokens.tolist()])
            pbar.update(1)
X_emb = X_emb.cpu().numpy()

try:
    if args.dataset == 'yp':
        y = ['0' if l in ['0', '1'] else l for l in y]
        y = ['4' if l in ['3', '4'] else l for l in y]
        X_emb = X_emb[[l != '2' for l in y], :]
        y = [l for l in y if l != '2']

    if args.dataset == 'wp':
        topics = [['wp', 'sp', 'tt'], ['eu'], ['cw'], ['pm'], ['mp', 'ip'], ['pi', 'cc'], ['ot'], ['rf']]
        match = [[True if l in t else False for t in topics] for l in y]
        y = [m.index(True) if True in m else None for m in match]
        X_emb = X_emb[[l is not None for l in y], :]
        y = [l for l in y if l is not None]

    if args.dataset == 'wi':
        X_emb = X_emb[[l is not None for l in y], :]
        y = [l for l in y if l is not None]

    # to 2D
    # X_emb_2d = TSNE(n_components=2, init='pca', verbose=1).fit_transform(X_emb)
    X_emb_2d = TSNE(n_components=2, verbose=1, perplexity=40).fit_transform(X_emb)


    def remove_outliers(data, r=2.0):
        outliers_data = abs(data - np.mean(data, axis=0)) >= r * np.std(data, axis=0)
        outliers = np.any(outliers_data, axis=1)
        keep = np.logical_not(outliers)
        return outliers, keep


    outliers, keep = remove_outliers(X_emb_2d)
    X_emb_2d = X_emb_2d[keep, :]
    y = [l for l, k in zip(y, keep.tolist()) if k]

    # plot
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    cc = ['r', 'b', 'g', 'y', 'k', 'c', 'm', 'tab:blue']
    for i, l in enumerate(sorted(set(y))):
        idx = [yl == l for yl in y]
        plt.scatter(X_emb_2d[idx, 0], X_emb_2d[idx, 1], c=cc[i], s=10, edgecolor='none', alpha=0.5)
    ax.axis('off')  # adding it will get no axis
    plt.savefig(os.path.join(save_folder, 'tSNE.png'))
    plt.close(fig)
except:
    pass

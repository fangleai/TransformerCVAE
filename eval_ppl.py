import pickle
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from tqdm import tqdm
from tqdm import trange
import importlib
import logging
import copy
from data.util import *
from util import *

from model import *


def compute_loss(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta):
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    mask = mask.to(device)
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    y_mask = y_mask.to(device)
    y_tokens = y_tokens.to(device)

    outputs = model(input_ids=input_tokens, attention_mask=mask, x_mask=x_mask, x_tokens=x_tokens, y_mask=y_mask,
                    y_tokens=y_tokens, from_prior=True)
    logits = outputs[0]
    kl_loss = outputs[-1]
    num_logits = logits.size(-1)

    # Perform masking
    if mask is not None:
        mask = mask.type(torch.bool)
        mask = mask.to(device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1)).mean()
    kl_loss = kl_loss.mean()
    loss = ce_loss + beta * kl_loss

    return loss, ce_loss, kl_loss


def compute_loss_ae(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta):
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    mask = mask.to(device)
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)

    outputs = model(input_ids=input_tokens, attention_mask=mask, y_mask=x_mask, y_tokens=x_tokens, from_mean=True, from_prior=False)

    logits = outputs[0]
    kl_loss = outputs[-1]
    num_logits = logits.size(-1)

    # Perform masking
    if mask is not None:
        mask = mask.type(torch.bool)
        mask = mask.to(device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1)).mean()
    kl_loss = kl_loss.mean()
    loss = ce_loss

    return loss, ce_loss, kl_loss


def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, help='pretrained model path to local checkpoint')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--out-dir', type=str, default='out')

    parser.add_argument('--data_type', type=str, default='t1', choices=['t' + str(i) for i in range(9)], help="t: type")
    parser.add_argument('--model_type', type=str, default='ae_vae_fusion', choices=['cvae', 'ae_vae_fusion'])
    parser.add_argument('--dataset', type=str, default='wi', choices=['wp', 'wi'], help="Dataset to use for training")
    parser.add_argument('--workers', default=1, type=int, metavar='N', help='number of data loading workers')

    # use GPU
    parser.add_argument('--gpu', default=3, type=int)
    parser.add_argument('--no_gpu', action="store_true")

    parser.add_argument('--fp16', action='store_true', help="Train using FP16?")

    parser.add_argument('--add_input', action="store_true")
    parser.add_argument('--add_attn', action="store_true")
    parser.add_argument('--add_softmax', action="store_true")
    parser.add_argument('--attn_proj_vary', action="store_true")

    parser.add_argument('--learn_prior', action="store_true")

    args = parser.parse_args('--model-path out/wi.2.proj_beta_half_ae/model_0150000.pt '
                             '--add_attn --learn_prior --fp16'.split())
    print(args)

    if args.model_type == 'cvae':
        args.learn_prior = True
    else:
        args.learn_prior = False

    # GPU
    if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu: torch.cuda.set_device(args.gpu)
    device = torch.device(args.gpu if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    if args.batch_size == -1:
        args.batch_size = 1

    # logging
    save_folder = args.model_path + '.eval/'
    os.makedirs(save_folder, exist_ok=True)
    importlib.reload(logging)
    logging.basicConfig(filename=os.path.join(save_folder, 'eval_ppl.log'),
                        level=logging.INFO, format='%(asctime)s--- %(message)s')
    logging.info('\n----------------------------------------------------------------------')

    print('Loading models...')
    cache_dir = os.path.join(args.out_dir, 'model_cache')
    os.makedirs(cache_dir, exist_ok=True)
    # Load pre-trained teacher tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
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

    VAE = VAEModel(config, add_input=args.add_input, add_attn=args.add_attn, add_softmax=args.add_softmax,
                   attn_proj_vary=args.attn_proj_vary, learn_prior=args.learn_prior)
    init_para_frompretrained(VAE.transformer, gpt2_model.transformer, share_para=True)
    init_para_frompretrained(VAE.encoder, gpt2_model.transformer, share_para=False)
    if args.learn_prior:
        init_para_frompretrained(VAE.encoder_prior, VAE.encoder, share_para=True)
        VAE.encoder_prior.averageSelfAttention.attention_weights = VAE.encoder.averageSelfAttention.attention_weights
    VAE.lm_head.weight = gpt2_model.lm_head.weight
    if VAE.add_softmax:
        VAE.lm_head_rep = Conv1D(*gpt2_model.lm_head.weight.size())
        # VAE.lm_head_rep = LM_head_rep(*gpt2_model.lm_head.weight.size()[::-1])
    print('VAE_params:', num_params(VAE))  # 286694400
    args.load = args.model_path
    if args.load:
        print('Loading model weights...')
        state = torch.load(os.path.join(args.load), map_location='cpu')
        if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
            state_copy = copy.copy(state)
            keys = state_copy.keys()
            for k in keys:
                state[k.replace('module.', '')] = state.pop(k)
        VAE.load_state_dict(state)
        gc.collect()
    print('Model loaded.')

    print('Setup data...')
    seq_len = VAE.config.n_ctx
    train_loader, val_loader, test_loader = prepare_dataset(
        args.data_dir, args.dataset, tokenizer,
        1, seq_len, 1, seq_len, args.batch_size, seq_len,
        make_test=True,
        num_workers=args.workers, data_type=args.data_type
    )
    print('Done.')

    if args.fp16:
        VAE = VAE.half()
    VAE.eval()  # be careful about VAE.eval() vs VAE.train()
    VAE.to(device)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    logging.info('\n----------------------------------------------------------------------')
    logging.info("Testing loop. batches: %d" % len(test_loader))

    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    startofcond = tokenizer.convert_tokens_to_ids("<|startofcond|>")
    endofcond = tokenizer.convert_tokens_to_ids("<|endofcond|>")

    n_words_bpe = 0
    n_words = 0
    logp_sum = 0.0

    n_words_bpe_l = []
    n_words_l = []
    logp_sum_l = []

    stats = []
    # test_iter = iter(test_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(test_iter)
    with tqdm(total=len(test_loader)) as pbar:
        for i_test, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask) in enumerate(test_loader):

            with torch.no_grad():
                if args.model_type == 'cvae':
                    loss, ce_loss, kl_loss = compute_loss(device, VAE, x_mask, x_tokens, y_mask, y_tokens, input_tokens,
                                                          target_tokens, mask, loss_fn, 1.0)
                else:
                    loss, ce_loss, kl_loss = compute_loss_ae(device, VAE, x_mask, x_tokens, y_mask, y_tokens, input_tokens,
                                                          target_tokens, mask, loss_fn, 1.0)

            stats.append([ce_loss.item(), math.exp(min(ce_loss.item(), 100)), kl_loss.item()])

            if len(target_tokens.size()) == 1:
               target_tokens = target_tokens.unsqueeze(0)
            n, l = target_tokens.size()

            tokens = target_tokens.tolist()
            tokens = [t[:t.index(endoftext) + 1] if endoftext in t else t for t in tokens]
            words_bpe = sum([len(t) for t in tokens])
            n_words_bpe += words_bpe
            n_words_bpe_l.append(words_bpe)

            story = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
            story = [s[:s.find("<|endoftext|>") + len("<|endoftext|>")] if "<|endoftext|>" in s else s for s in story]
            words = sum([len([t for t in re.split('("|\'|!|\?|\.|,|:| |\n|’|“|”|;|\(|\)|`)', s) if t != ' ' and t != '']) for s in story])
            n_words += words
            n_words_l.append(words)

            logp_sum += ce_loss.item() * words_bpe
            logp_sum_l.append(ce_loss.item() * words_bpe)

            #logging.info('test sample %05d finished.', i_test)
            pbar.update(1)

    print('Test complete with %05d samples.' % len(test_loader))
    logging.info("Test complete with %05d samples.", len(test_loader))

    print(' loss_bpe :', logp_sum / n_words_bpe)
    logging.info('loss_bpe: %f', logp_sum / n_words_bpe)

    ppl_bpe = round(math.exp(logp_sum / n_words_bpe), 3)
    ppl_word = round(math.exp(logp_sum / n_words), 3)
    print(' ppl_word:', ppl_word)
    print(' ppl_bpe :', ppl_bpe)
    logging.info('logp_sum: %f', logp_sum)
    logging.info('n_words_bpe: %d', n_words_bpe)
    logging.info('n_words    : %d', n_words)
    logging.info('    ppl_bpe : %f', ppl_bpe)
    logging.info('    ppl_word: %f', ppl_word)

    stats = np.mean(stats, axis=0)
    print(stats)
    logging.info('    stats: %s', str(stats))


if __name__ == '__main__':
    run_model()

import pickle
import os
import math
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
import numpy as np
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from tqdm import tqdm
from tqdm import trange
import importlib
import logging
import copy
from data.util import *
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
from util import *

from model import *


def top_k_top_p_filtering(logits, top_k=100, top_p=0.95, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def repeat_score(text, ngram=[3, 4, 5, 6]):
    ngram_list = []
    for ng in ngram:
        ngram_list.append([text[idx:idx + ng] for idx in range(len(text) - ng - 1)])

    max_occurs = []
    for ngrams in ngram_list:
        count_result = Counter([' '.join(n) for n in ngrams])
        try:
            max_occurs.append(
                max(count_result.values())
            )
        except:
            pass

    scores = [max_oc / ((len(text) / ngram[idx]) + ngram[idx]) for idx, max_oc in enumerate(max_occurs)]
    return max(scores) if len(scores) >= 1 else 1.0


def sample_sequence(model, tokenizer, length, batch_size=None, x_mask=None, x_tokens=None, y_mask=None, y_tokens=None,
                    temperature=1, top_k=100, top_p=0.95, device='cuda', sample=True, eos_token=None, model_type='cvae'):
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    y_mask = y_mask.to(device)
    y_tokens = y_tokens.to(device)

    mem = None
    prev = torch.tensor([[eos_token]] * batch_size, dtype=torch.long, device=device)

    output = prev
    probability = torch.tensor([], dtype=torch.float, device=device)
    if_end = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        if model_type == 'cvae':
            try:
                prior_mean, prior_logvar = model.encoder_prior(input_ids=x_tokens, attention_mask=x_mask)[:2]
            except:
                prior_mean = prior_logvar = torch.zeros([batch_size, model.config.n_embd], device=device)
            latent_mean, latent_logvar = prior_mean, prior_logvar
            z = model.reparameterize(latent_mean, latent_logvar)
            assert not torch.isnan(z).any(), 'training get nan z'
        else:
            posterior_mean, posterior_logvar = model.encoder(input_ids=x_tokens, attention_mask=x_mask)[:2]
            latent_mean, latent_logvar = posterior_mean, posterior_logvar
            z = latent_mean
            assert not torch.isnan(z).any(), 'training get nan z'

        for i in range(length): #trange
            logits, mem = model.transformer(input_ids=prev, past=mem, representations=z)

            logits = model.lm_head(logits)
            if model.add_softmax:
                logits_rep = model.lm_head_rep(z)
                logits = logits + logits_rep.unsqueeze(dim=1)

            logits = logits[:, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            if sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)

            probability = torch.cat((probability, probs.gather(1, next_token)), dim=1)
            output = torch.cat((output, next_token), dim=1)
            prev = next_token

            # early stopping if all sents have ended once
            if_end[next_token.view(-1).eq(eos_token)] = True
            if if_end.all(): break
    return output, probability


def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, help='pretrained model path to local checkpoint')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=int, default=0.95)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--out-dir', type=str, default='out')

    parser.add_argument('--data_type', type=str, default='t1', choices=['t' + str(i) for i in range(9)], help="t: type")
    parser.add_argument('--model_type', type=str, default='cvae', choices=['cvae', 'ae_vae_fusion'])
    parser.add_argument('--dataset', type=str, default='wi', choices=['wp', 'wi'], help="Dataset to use for training")

    # use GPU
    parser.add_argument('--gpu', default=2, type=int)
    parser.add_argument('--no_gpu', action="store_true")

    parser.add_argument('--add_input', action="store_true")
    parser.add_argument('--add_attn', action="store_true")
    parser.add_argument('--add_softmax', action="store_true")
    parser.add_argument('--attn_proj_vary', action="store_true")

    parser.add_argument('--learn_prior', action="store_true")

    args = parser.parse_args('--model-path out/wi.1.proj_vary_cyc_cvae/model_0030000.pt '
                             '--add_input --learn_prior '.split())
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
    if gpu: torch.cuda.manual_seed(args.seed)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    # logging
    save_folder = args.model_path + '.eval/'
    os.makedirs(save_folder, exist_ok=True)
    importlib.reload(logging)
    logging.basicConfig(filename=os.path.join(save_folder, 'eval.log'),
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

    # # add special tokens
    # special_tokens_dict = {
    #     'pad_token': '<|startoftext|>',
    #     'cls_token': '<|startofcond|>',
    #     'sep_token': '<|sepofcond|>',
    #     'mask_token': '<|endofcond|>'
    # }
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # print('We have added', num_added_toks, 'special tokens')
    # # Notice: resize_token_embeddings expect to receive the full size of the new vocab
    # gpt2_model.resize_token_embeddings(len(tokenizer))
    # assert tokenizer.pad_token == '<|startoftext|>'

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
    test_loader = prepare_dataset(
        args.data_dir, args.dataset, tokenizer,
        1, seq_len, 1, seq_len, args.batch_size, seq_len,
        make_train=False, make_val=False, make_test=True, data_type=args.data_type
    )[0]
    print('Done.')

    VAE.eval()  # be careful about VAE.eval() vs VAE.train()
    VAE.to(device)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    logging.info('\n----------------------------------------------------------------------')
    logging.info("Testing loop. batches: %d" % len(test_loader))

    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    startofcond = tokenizer.convert_tokens_to_ids("<|startofcond|>")
    endofcond = tokenizer.convert_tokens_to_ids("<|endofcond|>")

    n_samples = 0
    bleu4_sum = 0.0
    rouge_scores_values_sum = [0.0] * 9

    model_type = args.model_type

    # test_iter = iter(test_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(test_iter)
    with tqdm(total=len(test_loader)) as pbar:
        for i_test, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask) in enumerate(test_loader):

            length = args.length
            if length == -1:
                length = VAE.config.n_ctx - 1
            elif length > VAE.config.n_ctx - 1:
                raise ValueError("Can't get samples longer than window size: %s" % VAE.config.n_ctx)

            eff_samples = []
            n, l = target_tokens.size()
            storys = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
            storys_str = [s[:s.find("<|endoftext|>") + len("<|endoftext|>")] if "<|endoftext|>" in s else s for s in storys]

            for _ in range(args.nsamples // args.batch_size):
                # model, batch_size, temperature, top_k, top_p, eos_token, sample = VAE, args.batch_size, args.temperature, args.top_k, args.top_p, tokenizer.encoder['<|endoftext|>'], True
                out, _ = sample_sequence(
                    model=VAE,
                    tokenizer=tokenizer,
                    length=length,
                    batch_size=args.batch_size,
                    x_mask=x_mask,
                    x_tokens=x_tokens,
                    y_mask=y_mask,
                    y_tokens=y_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device = device,
                    eos_token=tokenizer.encoder['<|endoftext|>'],
                    model_type=model_type
                )
                out = out.tolist()

                # extract story, check metrics
                for i in range(len(out)):
                    text = out[i]
                    text = text[text.index(endoftext) + 1:]

                    if endoftext in text:
                        idx = text.index(endoftext)
                        text = text[:idx]

                    text = tokenizer.decode(text).strip()

                    # score for one long text, higher than 0.075 usually means repetition
                    # rep_score = repeat_score(text.split(), ngram=[3, 4, 5, 6, 7, 8])
                    # if rep_score > 0.075:
                    #     # print(rep_score)
                    #     continue

                    try:
                        # check bleu
                        bleu4 = sentence_bleu([storys_str[i].split()], text, smoothing_function=SmoothingFunction().method7)

                        # check rouge
                        rouge = Rouge()
                        rouge_scores = rouge.get_scores(text, storys_str[i])
                        rouge_scores_values = [v for k in rouge_scores[0].keys() for v in rouge_scores[0][k].values()]

                        bleu4_sum += bleu4
                        rouge_scores_values_sum = [v1 + v2 for v1, v2 in zip(rouge_scores_values_sum, rouge_scores_values)]
                        n_samples += 1
                    except:
                        bleu4 = 0.0
                        rouge_scores = [{'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                                         'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                                         'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}]

                    eff_samples.append((text, bleu4, rouge_scores))

                # write samples to file
                samples_file = open(save_folder + 'batch-' + '%04d' % i_test + '.txt', 'w', encoding='utf8')
                for i in range(len(eff_samples)):
                    samples_file.write("=" * 50 + " SAMPLE " + str(i) + " " + "=" * 50)
                    samples_file.write('\n' * 2)

                    samples_file.write("=" * 40 + " Outlines  " + "=" * 40)
                    samples_file.write('\n' * 2)
                    samples_file.write(tokenizer.decode(x_tokens[i, :][x_mask[i, :] == 1].tolist()))
                    samples_file.write('\n' * 2)
                    samples_file.write("=" * 40 + " Story " + "=" * 40)
                    samples_file.write('\n' * 2)
                    samples_file.write(storys_str[i])
                    samples_file.write('\n' * 2)

                    samples_file.write("=" * 40 + " Generated " + "=" * 40)
                    samples_file.write('\n' * 2)
                    samples_file.write(eff_samples[i][0])
                    samples_file.write('\n' * 4)
                    samples_file.flush()

                logging.info('batch %04d finished.', i_test)
                pbar.update(1)

    print('Test complete with %05d samples.' % n_samples)
    logging.info("Test complete with %05d samples.", n_samples)

    bleu4 = round(bleu4_sum / n_samples, 3)
    rouge_scores_values = [round(r / n_samples, 3) for r in rouge_scores_values_sum]
    print(' bleu-4:', bleu4)
    print(' rouge :', rouge_scores_values)
    logging.info(' bleu-4: %f', bleu4)
    logging.info(' rouge : %s', str(rouge_scores_values))


if __name__ == '__main__':
    run_model()

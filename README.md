# TransformerCVAE

This repository contains source code for paper [Transformer-based Conditional Variational Autoencoder for Controllable Story Generation](https://arxiv.org/abs/2101.00828):

```
@article{fang2021transformer,
  title={Transformer-based Conditional Variational Autoencoder for Controllable Story Generation},
  author={Fang, Le and Zeng, Tao and Liu, Chaochun and Bo, Liefeng and Dong, Wen and Chen, Changyou},
  journal={arXiv preprint arXiv:2101.00828},
  year={2021}
}
```

0. get source data ([Arxiv](https://github.com/gcunhase/ArXivAbsTitleDataset), [Yelp](https://github.com/fangleai/Implicit-LVM/tree/master/lang_model_yelp/data), [WritingPrompts](https://github.com/pytorch/fairseq/blob/master/examples/stories/README.md), [WikiPlots](https://github.com/markriedl/WikiPlots)).
1. data pre-processing (data/).
2. training (choose from several different implementations on parallelism and precision: train.py, train_dist.py, train_dist_half.py).
3. generation, evaluation and analysis (generate.py/generate_prefix.py, eval_ppl.py/eval_ppl_prefix.py, tsne_plot.py).

Contact: lefang@buffalo.edu

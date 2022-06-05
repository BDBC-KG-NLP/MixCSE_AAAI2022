# MixCSE_AAAI2022

A PyTorch implementation for our paper  "Unsupervised Sentence Representation via Contrastive Learning with Mixing Negatives".

You can download the paper from [here](https://www.aaai.org/AAAI22Papers/AAAI-8081.ZhangY.pdf).


## Abstract

Unsupervised sentence representation learning is a fundamental problem in natural language processing. Recently, contrastive learning has made great success on this task. Existing constrastive learning based models usually apply random sampling to select negative examples for training. Previous work in computer vision has shown that hard negative examples help contrastive learning to achieve faster convergency and better optimization for representation learning. However, the importance of hard negatives in contrastive learning for sentence representation is yet to be explored. In this study, we prove that hard negatives are essential for maintaining strong gradient signals in the training process while random sampling negative examples is ineffective for sentence representation. Accordingly, we present a contrastive model, MixCSE, that extends the current state-of-the-art SimCSE by continually constructing hard negatives via mixing both positive and negative features. The superior performance of the proposed approach is demonstrated via empirical studies on Semantic Textual Similarity datasets and Transfer task datasets

## Requirement

- Python = 3.7
- torch = 1.11.0
- numpy = 1.17.2
- transformers = 4.19.2



## train

```sh
bash run_unsup_example.sh
```

## evaluate
```
python evaluation.py \
    --model_name_or_path trained_model \
    --pooler cls \
    --task_set sts \
    --mode test
```

## Citation

If this work is helpful, please cite as:

```bibtex

@article{zhang2022unsupervised,
  title={Unsupervised Sentence Representation via Contrastive Learning with Mixing Negatives},
  author={Zhang, Yanzhao and Zhang, Richong and Mensah, Samuel and Liu, Xudong and Mao, Yongyi},
  year={2022}
}
```


## License

MIT

# MixCSE_AAAI2022

A PyTorch implementation for our paper  "Unsupervised Sentence Representation via Contrastive Learning with Mixing Negatives".

You can download the paper from [here](https://www.aaai.org/AAAI22Papers/AAAI-8081.ZhangY.pdf).


## Abstract



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

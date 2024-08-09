# [ACL 2024 Main] Controlled Text Generation for Black-box Language Models via Score-based Progressive Editor

This repository is the official implementation of Controlled Text Generation for Black-box Language Models via Score-based Progressive Editor, published as a main paper at ACL 2024.

https://arxiv.org/abs/2311.07430

In this version, only setting for Amazon-camera review corpus is available. Other settings will be available soon!

Our code is based on [Fairseq](https://github.com/facebookresearch/fairseq).

## Requirements
python == 3.8 (observed error about importing fairseq in python version >= 3.10)

openai

transformers

torch

mauve-text

## Download Links
you can download raw data of Amazon camera review corpus, roberta-base model, ScoPE checkpoint file at below link:

[Download_link](https://drive.google.com/drive/folders/1JNbESAlh3ZtK_0DgyGXHsb2YmZm8LS6O?usp=sharing)

## Install fairseq custom version
```
pip install –e ./fairseq_custom
```

## Construct training set from raw data
You should type your HuggingFace token id to the code.
```
CUDA_VISIBLE_DEVICES=0 python construct_training_set.py
```

### Preprocess training set for fairseq training process of ScoPE training
```
fairseq-preprocess --only-source --srcdict gpt2_bpe/dict.txt --trainpref amazon-raw/camera/camera.train.gpt2xl --validpref amazon-raw/camera/camera.valid.gpt2xl --testpref amazon-raw/camera/camera.valid.gpt2xl --destdir data-bin/camera-gpt2xl --workers 60
```

### Preprocess training set for fairseq training process of RoBERTa fine-tuning
```
fairseq-preprocess --only-source --srcdict gpt2_bpe/dict.txt --trainpref amazon-raw/camera/camera.train.gpt_block --validpref amazon-raw/camera/camera.valid.gpt_block --testpref amazon-raw/camera/camera.valid.gpt_block --destdir data-bin/roberta_ft_camera --workers 60
```

## Fine-tuning RoBERTa (w/ 1 GPU)
```
CUDA_VISIBLE_DEVICES=1 fairseq-train --task masked_lm --left-pad data-bin/roberta_ft_camera --save-dir checkpoints/roberta_ft_camera --restore-file roberta/roberta.base/model.pt --reset-optimizer --reset-dataloader --reset-meters --arch roberta_base --criterion masked_lm --dropout 0.1 --attention-dropout 0.1 --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --seed 1 --weight-decay 0.01 --clip-norm 0.0 --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --no-progress-bar --log-interval 100 --lr 1e-4 --lr-scheduler polynomial_decay --warmup-updates 1350 --tokens-per-sample 512 --sample-break-mode eos --max-tokens 16384 --update-freq 8 --total-num-update 13500 --max-update 13500 --fp16 --tensorboard-logdir checkpoints/roberta_ft_camera
```

### Convert fairseq Roberta to hf Roberta
Before convert, you should move dict.txt to checkpoint folder and rename the checkpoint file
```
cp gpt2_bpe/dict.txt checkpoints/roberta_ft_camera/dict.txt
cp checkpoints/roberta_ft_camera/checkpoint_last.pt checkpoints/roberta_ft_camera/model.pt
```

```
python convert_roberta_fairseq_to_hf.py --roberta_checkpoint_path checkpoints/roberta_ft_camera --pytorch_dump_folder_path roberta_ft_camera_hf
```


## Training ScoPE (w/ 4 GPUs)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train --task masked_lm --left-pad data-bin/camera-gpt2xl --save-dir checkpoints/scope_camera_review_gpt2xl --restore-file checkpoints/roberta_ft_camera/model.pt --reset-optimizer --reset-dataloader --reset-meters --arch roberta_base --criterion scope_loss --target-domain camera --dropout 0.1 --attention-dropout 0.1 --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --seed 1 --weight-decay 0.01 --clip-norm 0.0 --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --no-progress-bar --log-interval 100 --lr 1e-5 --lr-scheduler polynomial_decay --warmup-updates 4500 --tokens-per-sample 512 --sample-break-mode eos --max-tokens 2048 --update-freq 8 --total-num-update 45000 --max-update 45000 --fp16 --tensorboard-logdir checkpoints/scope_camera_review_gpt2xl
```

## ScoPE generation
Before generation, you should move dict.txt to checkpoint folder
```
cp gpt2_bpe/dict.txt checkpoints/scope_camera_review_gpt2xl /dict.txt
```

```
CUDA_VISIBLE_DEVICES=0 python generate_scope_openai_completion.py
```

## Citation
If you find this work useful, or you utilize it in your research, please cite the following paper:
```
@misc{yu2023ScoPE,
    title={Controlled Text Generation for Black-box Language Models via Score-based Progressive Editor},
    author={Sangwon Yu and Changmin Lee and Hojin Lee and Sungroh Yoon},
    year={2023},
    eprint={2311.07430},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

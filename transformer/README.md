# transformer

## 参考

- Transformer 代码参考 [nn.Transformer](https://github.com/pytorch/pytorch/blob/v1.12.1/torch/nn/modules/transformer.py) (torch==1.12.1)
- 机器翻译项目参考 [PyTorch 官方教程](https://pytorch.org/tutorials/beginner/translation_transformer.html)


## 安装环境

```
conda create -n transformer python=3.9
conda activate transformer

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

pip install torchtext==0.13.1
pip install torchdata==0.4.1
pip install spacy

python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## 运行代码
```
python language_translation.py
```
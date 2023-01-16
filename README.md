
## This code has been improved from informer
https://github.com/zhouhaoyi/Informer2020

## introduce
LSTF(Long time series forecasting)is of great prac-tical significance in many applications, which re-quires higher prediction accuracy and longer pre-diction range. With the recent emergence of Transformer, many LSTF methods have shifted their research focus to transformer, and effective progress has been made. However, the quadratic time complexity of Transformer leads to excessive computation overhead when faced with long se-quences. This prevents it from being applied di-rectly to the LSTF problem. To address these problems, we propose an Segmented Residual Self-attention architecture based on naive self-attention structure, bringing two promising features: (1) We propose to select segmented hy-perparameter to extend the self-attention method to a more universal form, making the computa-tional complexity adjustable and computational overhead  reducible (2)  By introducing residual shared parameters into the self-attention layer, a recurrent structure is constructed, In this way the self-attention layer can extract features well through segmented sequence, and reduce the over-fitting problem caused by  long sequence. Experi-mental results on various large-scale data sets show that our proposed method is significantly superior to the previous algorithm.

## Requirements

- Python 3.6
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.8.0

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data

The ETT dataset used in the paper can be downloaded in the repo [ETDataset](https://github.com/zhouhaoyi/ETDataset).
The required data files should be put into `data/ETT/` folder. A demo slice of the ETT data is illustrated in the following figure. Note that the input of each dataset is zero-mean normalized in this implementation.


The ECL data and Weather data can be downloaded here.
- [Google Drive](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR?usp=sharing)
- [BaiduPan](https://pan.baidu.com/s/1wyaGUisUICYHnfkZzWCwyA), password: 6gan 


## Usage

Commands for training and testing the model with *ProbSparse* self-attention on Dataset ETTh1, ETTh2 and ETTm1 respectively:

```bash
# ETTh1
python3 -u main.py --model srformer --data ETTh1 --freq h 

# ETTh2
python -u main.py --model srformer --data ETTh2  --freq h

# ETTm1
python -u main.py --model srformer --data ETTm1 --freq t
```



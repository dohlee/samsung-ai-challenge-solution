# Samsung AI Challenge solution

본 repository를 통해 2021년 DACON을 통해 개최된 [Samsung AI Challenge for Scientific Discovery](https://dacon.io/competitions/official/235789/overview/description) 경진대회의 5위 솔루션 코드를 정리하여 공개합니다.

## 1. 개요

본 챌린지에서는 분자의 3차원 구조 정보를 이용하여 S1-T1 사이의 에너지 갭을 추정할 수 있는 Machine Learning 알고리즘의 성능을 겨룹니다. 

## 2. 접근

### 입력 feature 설명

각 분자를 다음과 같이 featurize 하였습니다.

**원자 수준 feature**
- H, B, C, N, O, F, S, Si, P, Cl, Br, I 원자를 one-hot encoding (one-hot, 12-dim)
- Hybridization (one-hot, 7-dim)
- Aromatic ring 에 속한 atom인지 여부 (binary, 1-dim)
- 형식 전하 (1-dim)
- 결합된 수소 원자의 개수 (1-dim)
- 원자가 (1-dim)
- Donor/Acceptor status (one-hot, 2-dim)
- Spin multiplicity (one-hot, 2-dim)

**분자 수준 feature**
- NPR1 (Normalized principal moments ratio 1, `rdkit.Chem.Descriptors3D.NPR1` 활용)
- NPR2 (Normalized principal monents ratio 2, `rdkit.Chem.Descriptors3D.NPR2` 활용)

**분자의 3차원 구조 feature**
- 원자 쌍의 pairwise distance matrix (max_n_atoms x max_n_atoms)
- 원자 간 결합 정보를 나타내는 Adjacency matrix
- 결합각(bond angle) 및 이면각(torsion angle) 정보

### 모델 설명

분자의 원자 간 거리 행렬을 바탕으로 분자의 3차원 구조를 반영한 Transformer 모델을 사용하였습니다.

![model](/img/sac_model.png)

### 학습 방법

**Pretraining**
- 아래의 데이터셋을 이용하여 HOMO 및 LUMO를 예측하는 멀티태스크 사전학습을 수행합니다.
    - [QM9](http://quantum-machine.org/datasets/) (n=133,246)
    - [OE62](https://www.nature.com/articles/s41597-020-0385-y) (n=61,191)
- Pretraining에 사용되는 molecule sdf 데이터의 메타데이터(`pretrain_metadata.csv`)는 [여기](https://dohlee-bioinfo.sgp1.digitaloceanspaces.com/sac2021-data/pretrain_metadata.csv)에서 다운로드 받을 수 있습니다. 
- Pretraining을 위해 프로세싱 완료된 molecule sdf들을 모아 둔 디렉토리 `pretrain_sdf`는 [여기](https://dohlee-bioinfo.sgp1.digitaloceanspaces.com/sac2021-data/pretrain_sdf.tar.gz)에서 다운로드 받을 수 있습니다.

**Fine-tuning**
- 사전학습된 stem을 이용합니다.
    - 첫 9 epoch은 pretrained weight를 freeze 시킨 상태로 학습하고, 10 epoch 부터 weight unfreeze 후 모든 weight를 업데이트 시킵니다.
- 제공된 학습 데이터로 S1-T1 gap과, S1, T1 각각의 값을 예측하는 regression head를 학습합니다.
    - Gap, S1, T1 regression은 MSE loss를 사용합니다.
    - Gap의 weight는 1.0이고, S1, T1 regression의 weight는 0.05로 학습합니다.
- Optimizer = `AdamW(lr=3e-5)`
- Scheduler = `ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, threshold=0.005, threshold_mode='rel')`
- Batch size = 64

## 3. 설치 및 사용법
본 솔루션 코드 및 모델은 PyPI에 배포되어 있습니다. 모델은 다음과 같이 설치할 수 있습니다.
- 사용에 앞서서 `openbabel` 패키지가 환경에 설치되어 있어야 합니다. `conda install -c conda-forge openbabel`로 설치 가능합니다.

```bash
$ pip install sac2021
```

### Pretraining

- 메타데이터 [`pretrain_metadata.csv`](https://dohlee-bioinfo.sgp1.digitaloceanspaces.com/sac2021-data/pretrain_metadata.csv)와 sdf 파일 디렉토리 [`pretrain_sdf`](https://dohlee-bioinfo.sgp1.digitaloceanspaces.com/sac2021-data/pretrain_sdf.tar.gz)를 다운로드 후, 아래 명령의 `--meta`와 `--data` 파라미터를 적절히 설정하여 사전학습을 진행합니다.

```bash
$ python -m sac2021.pretrain \
    --meta [path/to/pretrain_metadata.csv] \
    --data [path/to/pretrain_sdf] \
    --output [OUTPUT_CHECKPOINT] \
    --model-id [ID] \
    --fold 0 \  # For validation purpose. (2.5% of the data will be held out)
    --loss mse \
```

Pretraining 학습 로그는 [이 Weight & Biases Project](https://wandb.ai/dohlee/sac-solution?workspace=user-dohlee)에서 확인 가능합니다.

### Fine-tuning
- 학습 데이터 `traindev.csv`와 sdf 파일 디렉토리 `traindev_sdf`를 다운로드 후, 아래 명령의 `--meta`와 `--data` 파라미터를 적절히 설정하여 fine-tuning을 진행합니다.
```bash
$ python -m sac2021.finetune \
    --meta [path/to/traindev.csv] \
    --data [path/to/traindev_sdf] \
    --ckpt [path/to/pretrained_checkpoint] \
    --output [OUTPUT_CHECKPOINT] \ 
    --model-id [ID] \ 
    --fold 0 \
    --loss mse
```

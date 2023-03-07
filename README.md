# Samsung AI Challenge solution

본 repository를 통해 2021년 DACON을 통해 개최된 [Samsung AI Challenge for Scientific Discovery](https://dacon.io/competitions/official/235789/overview/description) 경진대회의 5위 솔루션 코드를 정리하여 공개합니다.

## 1. 개요

WIP

## 2. 접근 방법 및 모델 설명

### 학습 방법

**Pretraining**
- 아래의 데이터셋을 이용하여 HOMO 및 LUMO를 예측하는 멀티태스크 사전학습을 수행합니다.
    - [QM9](http://quantum-machine.org/datasets/) (n=133,246)
    - [OE62](https://www.nature.com/articles/s41597-020-0385-y) (n=61,191)
- Pretraining에 사용한 molecule sdf 데이터는 [여기](https://dohlee-bioinfo.sgp1.digitaloceanspaces.com/sac2021-data%2Fpretrain_sdf.tar.gz)에서 다운로드 받을 수 있습니다.

**Fine-tuning**
- 사전학습된 stem을 이용합니다.
    - 첫 9 epoch은 pretrained weight를 freeze 시킨 상태로 학습하고, 10 epoch 부터 weight unfreeze 후 모든 weight를 업데이트 시킵니다.
- 제공된 학습 데이터로 S1-T1 gap과, S1, T1 각각의 값을 예측하는 regression head를 학습합니다.
    - Gap, S1, T1 regression은 MAE loss를 사용합니다.
    - Gap의 weight는 1.0이고, S1, T1 regression의 weight는 0.05로 학습합니다.
- Optimizer = `AdamW(lr=3e-5)`
- Scheduler = `ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, threshold=0.005, threshold_mode='rel')`
- Batch size = 64

## 3. 설치 및 사용법
본 솔루션 코드 및 모델은 PyPI에 배포되어 있습니다. 다음과 같이 설치할 수 있습니다.
```bash
$ pip install sac2021
```

# A Computer Vision Approach to Tracking Boxing

## Set-up

```bash
pip install -r requirements.txt
```

## To run

```bash
python main.py
```

## To train

```bash
cd compubox/
python3 compubox/models/train/train_hitnet.py --d /path/to/dataset --epochs <num epochs> --hit-keyword land --loss-every 1 --learning_rate <desired learning rate>
```

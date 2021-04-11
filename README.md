# Coreset AL
A greedy implementation of coreset based active learning for image classification (https://arxiv.org/abs/1708.00489)

How to run:

```
python active_learn.py \
                      --epochs $EPOCHS --lr $LR \
                      --gamma $GAMMA --init-size $INIT_SIZE \
                      --al-batch-size $AL_BSIZE \
                      --sampling-method $SAMPLE_METHOD \
                      --dataset-root $DROOT \
                      --dataset-name $DNAME \
                      --output-dir $OUT_DIR \
                      --max-eps $MAX_EPISODES
```

Check ```run_active_learn.sh``` for more details. 

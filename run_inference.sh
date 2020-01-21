DROOT=data/mnist_easy
DNAME=mnist
MODEL_FILE=output/mnist/202012111519/init.pth
CUDA_VISIBLE_DEVICES=0 python inference.py \
                      --dataset-root $DROOT \
                      --dataset-name $DNAME \
                      --model-file $MODEL_FILE

export CUDA_VISIBLE_DEVICES=2

python -m train_cifar10 --num_epochs 2000 --lr 0.001 --seed 18 --weight_decay 0.02 --exp_name lra_cifar10_lily --model_str LIN,160@TRAN,160,4,256,0.3,0.1,0.1,0.2,lily,hadamard,leakyrelu,LN,prenorm,8@REGH,10 --batch_size 16 --model_name "model" --accumulate 4
export CUDA_VISIBLE_DEVICES=1

python -m train_retrieval --num_epochs 2000 --lr 0.001 --seed 18 --weight_decay 0.01 --exp_name lra_retrieval_lily --model_str LIN,128@TRAN,128,4,256,0.0,0.0,0.1,0.2,lily,hadamard,leakyrelu,LN,postnorm,5@REGH,128 --batch_size 8 --model_name "model" --accumulate 4
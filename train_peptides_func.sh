export CUDA_VISIBLE_DEVICES=7

python -m train_peptides_func --num_epochs 3000 --lr 0.0008 --seed 18 --weight_decay 0.17 --exp_name peptide_func_lily --model_str LIN,144@TRAN,144,16,75,0.3,0.1,0.0,0.2,lily,none,leakyrelu,LN,postnorm,4@REGH,10 --batch_size 32 --model_name "model" --accumulate 1
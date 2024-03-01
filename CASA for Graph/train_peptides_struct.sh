export CUDA_VISIBLE_DEVICES=4

python -m train_peptides_struct --num_epochs 3000 --lr 0.001 --seed 18 --weight_decay 0.1 --exp_name peptide_struct --model_str LIN,256@TRAN,256,8,40,0.1,maxout,cat,leakyrelu,8@REGH,11,256,40,0.1 --batch_size 64 --model_name "model" --accumulate 1
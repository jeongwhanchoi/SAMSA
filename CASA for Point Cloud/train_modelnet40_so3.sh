export CUDA_VISIBLE_DEVICES=4

python -m train_modelnet40 --num_epochs 3000 --num_points 1024 --lr 0.001 --seed 18 --weight_decay 0.1 --input_mask True --exp_name modelnet_so3_softmax --uniform True --use_normal False --model_str LIN,256@TRAN,256,8,256,0.1,maxout,field,leakysoftmax,8@CLFH,40,256,512,0.1 --batch_size 64 --model_name "model" --accumulate 1
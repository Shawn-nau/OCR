
model_name=MLPQR
loss=Omni_cost_loss

root_path_name./Data/
data_path_name=processed_df.csv
save_path_name=./Exp_1/
random_seed=20241002

for set in 0 1 2 3 4 5 6 7 8 9 
do
python -u run_exp.py \
    --random_seed $random_seed \
    --checkpoints $save_path_name \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model $model_name \
    --loss $loss \
    --dim_embedding 3 \
    --hidden_size1 88 \
    --hidden_size2 44 \
    --train_epochs 100 \
    --s $set\
    --itr 1 --batch_size 5120 --learning_rate 0.001
done 




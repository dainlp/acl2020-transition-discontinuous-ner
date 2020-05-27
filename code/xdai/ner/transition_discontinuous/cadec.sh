for seed in 52 869 1001 50542 353778
do
  python train.py --output_dir /data/dai031/Experiments/TransitionDiscontinuous/cadec/$seed \
  --train_filepath /data/dai031/Experiments/CADEC/adr/split/train.txt \
  --dev_filepath /data/dai031/Experiments/CADEC/adr/split/dev.txt \
  --test_filepath /data/dai031/Experiments/CADEC/adr/split/test.txt \
  --log_filepath /data/dai031/Experiments/TransitionDiscontinuous/cadec/$seed/train.log \
  --model_type elmo --pretrained_model_dir /data/dai031/Corpora/ELMo/elmo_2x4096_512_2048cnn_2xhighway_5.5B \
  --weight_decay 0.0001 --max_grad_norm 5 \
  --learning_rate 0.001 --num_train_epochs 20 --patience 5 --eval_metric f1-overall \
  --max_save_checkpoints 0 \
  --cuda_device 0 --seed $seed
done
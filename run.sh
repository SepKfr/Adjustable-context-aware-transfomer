for dataset in watershed electricity traffic
do
  for seed in 21 9 1992
  do
    python train.py --seed $seed --exp_name $dataset
    python test.py --seed $seed --exp_name $dataset
  done
done

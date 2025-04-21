model=sft-gpt2
step=189696
device=0
split=test
model_dir=""
model_path=$model_dir/$model/step-$step/policy.pt
output_dir=out/$model/step-$step/split-$split

CUDA_VISIBLE_DEVICES=$device python evaluate.py --model $model_patn --output_dir $output_dir --datasets Binwei01/mmoqa_usa --n_examples 1843 --split $split
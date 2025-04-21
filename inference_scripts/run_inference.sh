# Run inference on the test dataset
model_dir=""
model=dpo-majority
step=14976
sampling_times=8
gpu_num=8
thread=$((sampling_times/gpu_num))
split=test
output_dir=out/$model/step-$step
mkdir -p $output_dir

for i in {0..7}
do
CUDA_VISIBLE_DEVICES=$i python inference.py --begin $((i*thread)) --end $((i*thread+thread)) --model gpt2-large --ckpt $model_dir/$model/step-$step/policy.pt --output_dir $output_dir --datasets Binwei01/mmoqa_usa --n_examples 344 --max_length 1024 --max_prompt_length 1000 --split $split &
done

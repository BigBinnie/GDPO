output_dir=outputs/sft-gpt2
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py model=gpt2-large datasets=[Binwei01/mmoqa_usa] loss=sft exp_name=sft-gpt2 gradient_accumulation_steps=4 batch_size=32 eval_batch_size=16 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=10000 n_epochs=20 output_dir=$output_dir
# max_length=1024 max_prompt_length=1000

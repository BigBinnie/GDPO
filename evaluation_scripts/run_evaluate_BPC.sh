src_dir=out/sft-gpt2/step-189696
save_dir=$src_dir
python evaluate_BPC.py \
--src_dir ${src_dir} \
--save  ${save_dir} \
--b 0 \
--e 100000 \
--datasets Binwei01/mmoqa_usa \
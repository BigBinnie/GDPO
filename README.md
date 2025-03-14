# GDPO
This repository contains data and code for the paper: [No Preference Left Behind: Group Distributional Preference Optimization](http://arxiv.org/abs/2412.20299). This paper is accepted by ICLR 2025. If you have any questions, please reach out to the authors at binwei.yao@wisc.edu.
# Overview
Preferences within a group of people are not uniform but follow a distribution. While existing alignment methods like Direct Preference Optimization (DPO) attempt to steer models to reflect human preferences, they struggle to capture the distributional pluralistic preferences within a group. These methods often skew toward dominant preferences, overlooking the diversity of opinions, especially when conflicting preferences arise. To address this issue, we propose Group Distribution Preference Optimization (GDPO), a novel framework that aligns language models with the distribution of preferences within a group by incorporating the concept of beliefs that shape individual preferences. GDPO calibrates a language model using statistical estimation of the group's belief distribution and aligns the model with belief-conditioned preferences, offering a more inclusive alignment framework than traditional methods. In experiments using both synthetic controllable opinion generation and real-world movie review datasets, we show that DPO fails to align with the targeted belief distributions, while GDPO consistently reduces this alignment gap during training. Moreover, our evaluation metrics demonstrate that GDPO outperforms existing approaches in aligning with group distributional preferences, marking a significant advance in pluralistic alignment. 
# Data Synthesis
Dataset Statistics of Controllable Opinion Generation: The number following the country name is the sum of questions in GlobalOpinionQA used to generate dialogues.
| Split  | [United States (469) Small](https://huggingface.co/datasets/Binwei01/mmoqa_usa) | [United States (469) Large](https://huggingface.co/datasets/Binwei01/mmoqa_usa_large) | [Pakistan (219) Small](https://huggingface.co/datasets/Binwei01/mmoqa_pk) | [Pakistan (219) Large](https://huggingface.co/datasets/Binwei01/mmoqa_pk_large) | [S. Africa (162) Small](https://huggingface.co/datasets/Binwei01/mmoqa_sa) | [S. Africa (162) Large](https://huggingface.co/datasets/Binwei01/mmoqa_sa_large) |
|--------|----------------------------|----------------------------|----------------------|----------------------|----------------------|----------------------|
| Train  | 14,321                     | 176,905                    | 6,684                | 80,364               | 4,960                | 54,896               |
| Eval   | 1,843                      | 22,166                     | 860                  | 10,070               | 636                  | 6,878                |
| Test   | 1,843                      | 22,199                     | 876                  | 10,086               | 648                  | 6,890                |

# Alignment
Our experiments follow the implementation of [DPO](https://github.com/eric-mitchell/direct-preference-optimization).
## 1. Set up the experiment
``pip install -r requirements.txt``
## 2. SFT Training
Please set parameters of model, dataset and output_dir as necessary. You can directly use the dataset in Section 2. If you would like use your own dataset, please implement your dataset preprocessing method in ``preference_datasets.py``. 
``bash train_scripts/run_sft.sh``
## 3. DPO Training
Please set parameters of model, dataset, **sft_model_path** and output_dir as necessary.
``bash train_scripts/run_dpo.sh``
## 4. GDPO Training
Please set parameters of model, dataset, **sft_model_path** and output_dir as necessary.
``bash train_scripts/run_gdpo.sh``
# Evaluation

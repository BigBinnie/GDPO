# GDPO
This repository contains data and code for the paper: [No Preference Left Behind: Group Distributional Preference Optimization](http://arxiv.org/abs/2412.20299). This paper is accepted by ICLR 2025. If you have any questions, please reach out to the authors at binwei.yao@wisc.edu.
# Overview
Group preferences are diverse and follow a distribution rather than being uniform. Existing alignment methods like Direct Preference Optimization (DPO) struggle to capture this diversity, often favoring dominant preferences and overlooking conflicting ones. To address this, the authors propose Group Distributional Preference Optimization (GDPO), a novel framework that aligns language models with the distribution of preferences within a group by incorporating beliefs that shape individual preferences. GDPO estimates the group's belief distribution and aligns the model accordingly, ensuring a more inclusive approach. Experiments on synthetic and real-world datasets show that DPO fails to match targeted belief distributions, while GDPO significantly reduces this gap and outperforms existing methods in pluralistic alignment.
# Data Synthesis
1. Dataset Statistics of Controllable Opinion Generation: The number following the country name is the sum of questions in GlobalOpinionQA used to generate dialogues.

| Split  | [United States (469) Small](https://huggingface.co/datasets/Binwei01/mmoqa_usa) | [United States (469) Large](https://huggingface.co/datasets/Binwei01/mmoqa_usa_large) | [Pakistan (219) Small](https://huggingface.co/datasets/Binwei01/mmoqa_pk) | [Pakistan (219) Large](https://huggingface.co/datasets/Binwei01/mmoqa_pk_large) | [S. Africa (162) Small](https://huggingface.co/datasets/Binwei01/mmoqa_sa) | [S. Africa (162) Large](https://huggingface.co/datasets/Binwei01/mmoqa_sa_large) |
|--------|----------------------------|----------------------------|----------------------|----------------------|----------------------|----------------------|
| Train  | 14,321                     | 176,905                    | 6,684                | 80,364               | 4,960                | 54,896               |
| Eval   | 1,843                      | 22,166                     | 860                  | 10,070               | 636                  | 6,878                |
| Test   | 1,843                      | 22,199                     | 876                  | 10,086               | 648                  | 6,890                |

2. Dataset Statistics of Controllable Review Generation: The data is from [Amazon Movie Review](https://snap.stanford.edu/data/web-Movies.html).

| Split  | [Small](https://huggingface.co/datasets/Binwei01/movie_review)  | [Large](https://huggingface.co/datasets/Binwei01/movie_review_large)  |
|--------|--------|--------|
| **Train** | 13,825 | 73,804 |
| **Eval**  | 1,657  | 9,155  |
| **Test**  | 2,406  | 10,114 |

# Training Process
Our experiments follow the implementation of [DPO](https://github.com/eric-mitchell/direct-preference-optimization). The training has two steps: 1. supervised finetuning (SFT); 2. preference learning by DPO or GDPO.
## 1. Set up the experiment
``pip install -r requirements.txt``
## 2. SFT Training
Please set parameters of model, dataset and output_dir as necessary. You can directly use the dataset in Section 2. If you would like use your own dataset, please implement your dataset preprocessing method in ``preference_datasets.py``. 

``bash train_scripts/run_sft.sh``
## 3. DPO Training
Please set parameters of model, dataset, **sft_model_path** and output_dir as needed.

``bash train_scripts/run_dpo.sh``
## 4. GDPO Training
Please set parameters of model, dataset, **sft_model_path** and output_dir as needed.

``bash train_scripts/run_gdpo.sh``
# Evaluation
Please set parameters of model, dataset, and output_dir as needed.
``bash inference_scripts/run_inference.sh``
# Citation
If you would like to use our data or method in your paper, please cite:

    @article{yao2024no,
    title={No Preference Left Behind: Group Distributional Preference Optimization},
    author={Yao, Binwei and Cai, Zefan and Chuang, Yun-Shiuan and Yang, Shanglin and Jiang, Ming and Yang, Diyi and Hu, Junjie},
    journal={arXiv preprint arXiv:2412.20299},
    year={2024}
    }

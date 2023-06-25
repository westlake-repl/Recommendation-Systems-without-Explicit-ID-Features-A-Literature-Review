# Recommendation Systems without Explicit ID Features: A Literature Review

<!-- <font size=6><center><big><b> [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) </b></big></center></font> -->

Paper list of LLM (Large Language Model), multimodal, transferable and foundation Recommender System. 

**Keyword:** *Recommend System, pretraining, large language model, multimodal recommender system, transferable recommender system, foundation recommender models, ID features, ID embeddings*

**These papers attempt to address the following questions:** 

(1) Can recommender systems have their own foundation models similar to those used in NLP and CV? 

(2) Can explicit item ID embeddings be replaced or abandoned? 

(3) Will recommender systems shift from a matching paradigm to a generating paradigm?

(4) How can LLM be utilized to enhance recommender systems?

(5) What does the future hold for multimodal recommender systems?

**Welcome to open an issue or make a pull request!** 
# Paper List 

## Perspective Paper 
- Where to Go Next for Recommender Systems? ID-vs. Modality-based recommender models revisited, openreview 2022/09, [[paper]](https://arxiv.org/pdf/2303.13835.pdf)
- Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights, arxiv 2023/05, [[paper]](https://arxiv.org/pdf/2305.11700.pdf)


## Survey

## Large Language Models for Recommendation （LLM4Rec）
### Tuning LLM
- M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems,arxiv 2022/05, [[paper]](https://arxiv.org/pdf/2205.08084.pdf)
- TALLRec: An Effective and Efficient Tuning Framework to Align Large  Language Model with Recommendation, arxiv 2023/04, [[paper]](http://arxiv.org/abs/2305.00447v1)
- PTab: Using the Pre-trained Language Model for Modeling Tabular Data, arxiv 2022/09, [[paper]](https://arxiv.org/abs/2209.08060)
- Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5), Recsys 2022/05, [[paper]](https://arxiv.org/abs/2203.13366))
### Freezing LLM
- CTR-BERT: Cost-effective knowledge distillation for billion-parameter teacher models，arxiv 2022/04,  [[paper]](https://neurips2021-nlp.github.io/papers/20/CameraReady/camera_ready_final.pdf)
### Prompt with LLM
- Prompt Learning for News Recommendation, SIGIR 2023/04, [[paper]](https://arxiv.org/abs/2304.05263)
- Language Models as Recommender Systems: Evaluations and Limitations, NeurIPS Workshop ICBINB 2021/10, [[paper]](https://openreview.net/pdf?id=hFx3fY7-m9b)
### ChatGPT
- Is ChatGPT a Good Recommender A Preliminary Study, arxiv 2023/04, [[paper]](https://arxiv.org/pdf/2304.10149.pdf)
- Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent, arxiv 2023/04, [[paper]](https://arxiv.org/pdf/2304.09542.pdf)
- Large Language Models are Zero-Shot Rankers for Recommender Systems, arxiv 2023/05, [[paper]](http://arxiv.org/abs/2305.08845v1)
- Recommendation as Instruction Following: A Large Language Model  Empowered Recommendation Approach, arxiv 2023/05, [[paper]](http://arxiv.org/abs/2305.07001v1)
- Leveraging Large Language Models in Conversational Recommender Systems, arxiv 2023/05, [[paper]](http://arxiv.org/abs/2305.07961v2)
- Uncovering ChatGPT’s Capabilities in Recommender Systems, arxiv 2023/05, [[paper]](https://arxiv.org/pdf/2305.02182.pdf)[[code]](https://github.com/rainym00d/LLM4RS)
- Sparks of Artificial General Recommender (AGR): Early Experiments with ChatGPT, arxiv 2023/05, [[paper]](https://arxiv.org/pdf/2305.04518.pdf)
- Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation, arxiv 2023/05,[[paper]](https://arxiv.org/pdf/2305.07609.pdf)
[[code]](https://github.com/jizhi-zhang/FaiRLLM)












- PALR: Personalization Aware LLMs for Recommendation, arxiv 2023/05, [[paper]](http://arxiv.org/abs/2305.07622v1)

- Privacy-Preserving Recommender Systems with Synthetic Query Generation  using Differentially Private Large Language Models, arxiv 2023/05, [[paper]](http://arxiv.org/abs/2305.05973v1)

- CTRL: Connect Tabular and Language Model for CTR Prediction, arxiv 2023/06,[[paper]](https://arxiv.org/abs/2306.02841)
  
## Multimodal Recommender System 


## Foundation Recommender models
- Exploring Adapter-based Transfer Learning for Recommender Systems: Empirical Studies and Practical Insights,arxiv 2022, [[paper]](https://arxiv.org/pdf/2305.15036.pdf)

## Universal User Reresentation


## Generative Recommender Systems

## Abandoning Explicit ID Features

## Dataset
- Yelp[[link]](https://www.yelp.com/dataset)
- Petdata[[link]](https://drive.google.com/file/d/1OcvbBJN0jlPTEjE0lvcDfXRkzOjepMXH/view)
- M5Product: Self-harmonized Contrastive Learning for E-commercial Multi-modal Pretraining, CVPR 2022 [[paper]](https://arxiv.org/pdf/2109.04275.pdf)
- Tenrec: A Large-scale Multipurpose Benchmark Dataset for Recommender Systems, NeurIPS 2022  [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/4ad4fc1528374422dd7a69dea9e72948-Paper-Datasets_and_Benchmarks.pdf)
- Where to Go Next for Recommender Systems? ID-vs. Modality-based recommender models revisited, openreview 2022/09, [[paper]](https://arxiv.org/pdf/2303.13835.pdf)








<!-- ## Sequential / Session-Based Recommendation
- BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer, CIKM 2019 ,  [[paper]](https://arxiv.org/abs/1904.06690)[[code]](https://github.com/FeiSun/BERT4Rec)
- S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization
, CIKM-2020 , [[paper]](https://arxiv.org/abs/2008.07873)[[code]](https://github.com/RUCAIBox/CIKM2020-S3Rec)
- Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation, Recsys 2021
 , [[paper]](https://dl.acm.org/doi/abs/10.1145/3460231.3474255?casa_token=b4-oEoLXZycAAAAA:khQBoMBHAS5TXADNUar92RYFH4bq68KSjk3VvD5FDJzazv3jXXfcj_LHdnREjvfUgYj-4dipepKs)[[code]](https://github.com/NVIDIA-Merlin/Transformers4Rec)
 - Towards Universal Sequence Representation Learning for Recommender Systems , KDD 2022 , [[paper]](https://arxiv.org/pdf/2206.05941.pdf)[[code]](https://github.com/RUCAIBox/UniSRec)
- Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders, WWW 2023, [[paper]](https://arxiv.org/abs/2210.12316) [[code]](https://github.com/RUCAIBox/VQ-Rec) -->

<!-- ## User Representation Pretraining 
- Parameter-Efficient Transfer from Sequential Behaviors for User Modeling and Recommendation, SIGIR 2020 , [[paper]](https://arxiv.org/pdf/2001.04253.pdf), [[code]](https://github.com/fajieyuan/sigir2020_peterrec)
- UPRec: User-Aware Pre-training for Recommender Systems ，submitted TKDE in 2021 , [[paper]](https://arxiv.org/abs/2102.10989)
- U-BERT: Pre-training user representations for improved recommendation, AAAI 2021, [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16557)
- UserBERT: Self-supervised User Representation Learning ,  arxiv 2021 , [[paper]](https://arxiv.org/abs/2109.01274)
- One4all User Representation for Recommender Systems in E-commerce , arxiv 2021 , [[paper]](https://arxiv.org/abs/2106.00573)
- One Person, One Model, One World: Learning Continual User Representation without Forgetting, SIGIR 2021 , [[paper]](https://arxiv.org/pdf/2001.04253.pdf)
- Scaling Law for Recommendation Models: Towards General-purpose User Representations , AAAI 2023 , [[paper]](https://arxiv.org/abs/2111.11294) -->

<!-- ## Two Tower Pretraining
- Self-supervised Learning for Large-scale Item Recommendations , CIKM 2021 , [[paper]](https://dl.acm.org/doi/pdf/10.1145/3459637.3481952)
- TransRec: Learning Transferable Recommendation from Mixture-of-Modality Feedback , arxiv 2022 , [[paper]](https://arxiv.org/abs/2206.06190)
- IntTower: the Next Generation of Two-Tower Model for Pre-Ranking System, CIKM 2022 , [[paper]](https://arxiv.org/abs/2210.09890)[[code]](https://github.com/archersama/inttower) -->

<!-- ## Language Models as Recommenddation Models & Prompt Learning -->



# Workshop and Tutorial
-


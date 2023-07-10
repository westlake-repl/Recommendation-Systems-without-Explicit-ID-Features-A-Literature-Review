# Foundation models for Recommender System Paper List
**Welcome to open an issue or make a pull request!** 

<!-- <font size=6><center><big><b> [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) </b></big></center></font> -->


**Keyword:** *Recommend System, pretraining, large language model, multimodal recommender system, transferable recommender system, foundation recommender models, ID features, ID embeddings*

**These papers attempt to address the following questions:** 

(1) Can recommender systems have their own foundation models similar to those used in NLP and CV? 

(2) Is ID embedding necessary for recommender models, can we replace or abondon it? 

(3) Will recommender systems shift from a matching paradigm to a generating paradigm?

(4) How can LLM be utilized to enhance recommender systems?

(5) What does the future hold for multimodal recommender systems?


# Paper List 

## Perspective Paper (ID vs LLM & Multimodal)
- Where to Go Next for Recommender Systems? ID-vs. Modality-based recommender models revisited, SIGIR2023, 2022/09, [[paper]](https://arxiv.org/pdf/2303.13835.pdf)  [[code]](https://github.com/westlake-repl/IDvs.MoRec)
- Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights, arxiv 2023/05, [[paper]](https://arxiv.org/pdf/2305.11700.pdf)




## Survey
- How Can Recommender Systems Benefit from Large Language Models: A Survey, arxiv 2023/06, [[paper]](https://arxiv.org/abs/2306.05817)
- Recommender Systems in the Era of Large Language Models, arxiv, 2023/07, [[paper]](https://arxiv.org/pdf/2307.02046.pdf)
- A Survey on Large Language Models for Recommendation, arxiv 2023/05, [[paper]](https://arxiv.org/abs/2305.19860)

## Large Language Models for Recommendation （LLM4Rec）
### Tuning LLM
- M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems,arxiv 2022/05, [[paper]](https://arxiv.org/pdf/2205.08084.pdf) 
- Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5), Recsys 2022/05, [[paper]](https://arxiv.org/abs/2203.13366))
- PTab: Using the Pre-trained Language Model for Modeling Tabular Data, arxiv 2022/09, [[paper]](https://arxiv.org/abs/2209.08060)
- TALLRec: An Effective and Efficient Tuning Framework to Align Large  Language Model with Recommendation, arxiv 2023/04, [[paper]](http://arxiv.org/abs/2305.00447v1)
- GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation, 2023/04, [[paper]](https://arxiv.org/pdf/2304.03879.pdf)
### Freezing LLM [[link]](https://github.com/WLiK/LLM4Rec)
- CTR-BERT: Cost-effective knowledge distillation for billion-parameter teacher models，arxiv 2022/04,  [[paper]](https://neurips2021-nlp.github.io/papers/20/CameraReady/camera_ready_final.pdf)
- Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning, arxiv 2022/06, [[paper]](https://arxiv.org/abs/2206.09363)
- Generative Recommendation: Towards Next-generation Recommender Paradigm, arxiv 2023/04, [[paper]](https://arxiv.org/abd/2304.03516)
- Leveraging Large Language Models in ConversationalRecommender Systems, arxiv 2023/05, [[paper]](https://arxiv.org/abs/2305.07961)
- A First Look at LLM-Powered Generative News Recommendation, arxiv 2023/05, [[paper]](https://arxiv.org/abs/2305.06566)
- Privacy-Preserving Recommender Systems with Synthetic Query Generation using Differentially Private Large Language Models, arxiv 2023/05,[[paper]](https://arxiv.org/abs/2305.05973)
- RecAgent: A Novel Simulation Paradigm for Recommender Systems, arxiv 2023/06, [[paper]](https://arxiv.org/abs/2306.02552)
- Zero-Shot Next-Item Recommendation using Large Pretrained Language Models, arxiv 2023/04, [[paper]](https://arxiv.org/abs/2304.03153)
- Can ChatGPT Make Fair Recommendation? A Fairness Evaluation Benchmark for Recommendation with Large Language Model, RecSys 2023



### Prompt with LLM
- Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5), RecSys 2022, [[paper]](https://arxiv.org/pdf/2203.13366.pdf)
- Language Models as Recommender Systems: Evaluations and Limitations, NeurIPS Workshop ICBINB 2021/10, [[paper]](https://openreview.net/pdf?id=hFx3fY7-m9b)
- Prompt Learning for News Recommendation, SIGIR 2023/04, [[paper]](https://arxiv.org/abs/2304.05263)
### ChatGPT [[link]](https://github.com/archersama/awesome-recommend-system-pretraining-papers)
- Is ChatGPT a Good Recommender A Preliminary Study, arxiv 2023/04, [[paper]](https://arxiv.org/pdf/2304.10149.pdf)
- Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent, arxiv 2023/04, [[paper]](https://arxiv.org/pdf/2304.09542.pdf)
- Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System, arxiv 2023/04,[[paper]](https://arxiv.org/abs/2303.14524)
- Large Language Models are Zero-Shot Rankers for Recommender Systems, arxiv 2023/05, [[paper]](http://arxiv.org/abs/2305.08845v1)
- Recommendation as Instruction Following: A Large Language Model  Empowered Recommendation Approach, arxiv 2023/05, [[paper]](http://arxiv.org/abs/2305.07001v1)
- Leveraging Large Language Models in Conversational Recommender Systems, arxiv 2023/05, [[paper]](http://arxiv.org/abs/2305.07961v2)
- Uncovering ChatGPT’s Capabilities in Recommender Systems, arxiv 2023/05, [[paper]](https://arxiv.org/pdf/2305.02182.pdf)[[code]](https://github.com/rainym00d/LLM4RS)
- Sparks of Artificial General Recommender (AGR): Early Experiments with ChatGPT, arxiv 2023/05, [[paper]](https://arxiv.org/pdf/2305.04518.pdf)
- Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation, arxiv 2023/05,[[paper]](https://arxiv.org/pdf/2305.07609.pdf)
[[code]](https://github.com/jizhi-zhang/FaiRLLM)
- Sparks of Artificial General Recommender (AGR): Early Experiments with ChatGPT, arxiv 2023/05,[[paper]](https://arxiv.org/abs/2305.04518)
- PALR: Personalization Aware LLMs for Recommendation, arxiv 2023/05, [[paper]](http://arxiv.org/abs/2305.07622v1)
- Privacy-Preserving Recommender Systems with Synthetic Query Generation  using Differentially Private Large Language Models, arxiv 2023/05, [[paper]](http://arxiv.org/abs/2305.05973v1)
- Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models, arxiv 2023/05, [[paper]](https://arxiv.org/abs/2305.13112)
- CTRL: Connect Tabular and Language Model for CTR Prediction, arxiv 2023/06,[[paper]](https://arxiv.org/abs/2306.02841)


  
## Multimodal Recommender System 
- VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback, AAAI2016, [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/9973)
- Multi-modal Knowledge Graphs for Recommender Systems, CIKM 2020, [[paper]](https://zheng-kai.com/paper/cikm_2020_sun.pdf)


## Foundation and Transferable Recommender models
- TransRec: Learning Transferable Recommendation from Mixture-of-Modality Feedback, arxiv 2022/06, [[paper]](https://www.semanticscholar.org/paper/TransRec%3A-Learning-Transferable-Recommendation-from-Wang-Yuan/f7c9551e19fabf5d534115a5704a5f8fea097534)
- Towards Universal Sequence Representation Learning for Recommender Systems, KDD2022,2022/06, [[paper]](https://arxiv.org/pdf/2206.05941.pdf)
- Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders, WWW 2023, [[paper]](https://arxiv.org/abs/2210.12316) [[code]](https://github.com/RUCAIBox/VQ-Rec)
- Exploring Adapter-based Transfer Learning for Recommender Systems: Empirical Studies and Practical Insights, arxiv 2023, [[paper]](https://arxiv.org/pdf/2305.15036.pdf)
- UP5: Unbiased Foundation Model for Fairness-aware Recommendation, arxiv 2023/05, [[paper]](https://arxiv.org/pdf/2305.12090.pdf)
- VIP5: Towards Multimodal Foundation Models for Recommendation, arxiv 2023/05, [[paper]](https://arxiv.org/pdf/2305.12090.pdf)
- OpenP5: Benchmarking Foundation Models for Recommendation, arxiv 2023/06, [[paper]](https://arxiv.org/pdf/2306.11134.pdf)

## Universal One4all Representation Learning 
- Parameter-Efficient Transfer from Sequential Behaviors for User Modeling and Recommendation, SIGIR 2020, [[paper]](https://arxiv.org/pdf/2001.04253.pdf), [[code]](https://github.com/yuangh-x/2022-NIPS-Tenrec)
- U-BERT: Pre-training user representations for improved recommendation, AAAI 2021, [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16557)
- One4all User Representation for Recommender Systems in E-commerce, arxiv 2021, [[paper]](https://arxiv.org/abs/2106.00573)
- Learning Transferable User Representations with Sequential Behaviors via Contrastive Pre-training, ICDM 2021, [[paper]](https://fajieyuan.github.io/papers/ICDM2021.pdf)
- Scaling Law for Recommendation Models: Towards General-purpose User Representations, AAAI 2023, [[paper]](https://arxiv.org/abs/2111.11294)

## Lifelong User Representation Learning 
- One Person, One Model, One World: Learning Continual User Representation without Forgetting, SIGIR 2021, [[paper]](https://arxiv.org/pdf/2001.04253.pdf), [[code]](https://github.com/yuangh-x/2022-NIPS-Tenrec)
- Tenrec: A Large-scale Multipurpose Benchmark Dataset for Recommender Systems, NeurIPS 2022  [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/4ad4fc1528374422dd7a69dea9e72948-Paper-Datasets_and_Benchmarks.pdf)
- STAN: Stage-Adaptive Network for Multi-Task Recommendation by Learning User Lifecycle-Based Representationg, arxiv 2023,  [[paper]](https://arxiv.org/abs/2306.12232)
- Task Relation-aware Continual User Representation Learning, KDD2023,  [[paper]](https://arxiv.org/pdf/2306.01792.pdf)
  

## Generative Recommender Systems [[link]](https://github.com/gabriben/awesome-generative-information-retrieval#generative-recommendation)
- A Simple Convolutional Generative Network for Next Item Recommendation, WSDM 2018/08, [[paper]](https://arxiv.org/pdf/1808.05163.pdf)  [[code]](https://github.com/fajieyuan/WSDM2019-nextitnet)
- Future Data Helps Training: Modeling Future Contexts for Session-based Recommendation, WWW 2020/04, [[paper]](https://arxiv.org/pdf/1906.04473.pdf)  [[code]](https://github.com/fajieyuan/WWW2020-grec)
- Recommendation via Collaborative Diffusion Generative Model, KSEM 2022/08, [[paper]](https://dl.acm.org/doi/abs/10.1007/978-3-031-10989-8_47)
- Blurring-Sharpening Process Models for Collaborative Filtering, arxiv 2022/09, [[paper]](https://arxiv.org/abs/2211.09324)
- Generative Slate Recommendation with Reinforcement Learning, arxiv 2023/01, [[paper]](https://arxiv.org/abs/2301.08632)
- Recommender Systems with Generative Retrieval, arxiv 2023/04, [[paper]](https://arxiv.org/pdf/2305.05065.pdf)
- DiffuRec: A Diffusion Model for Sequential Recommendation, arxiv 2023/04, [[paper]](https://arxiv.org/abs/2304.00686)
- Diffusion Recommender Model, arxiv 2023/04, [[paper]](https://arxiv.org/abs/2304.04971)
- A First Look at LLM-Powered Generative News Recommendation, arxiv 2023/05, [[paper]](https://arxiv.org/abs/2305.06566)
- Recommender Systems with Generative Retrieval, arxiv 2023/05, [[paper]](https://arxiv.org/abs/2305.05065)
- Generative Retrieval as Dense Retrieval, arxiv 2023/06, [[paper]](https://arxiv.org/pdf/2306.11397.pdf)
- RecFusion: A Binomial Diffusion Process for 1D Data for Recommendation, arxiv 2023/06, [[paper]](https://arxiv.org/abs/2306.08947)
- Generative Sequential Recommendation with GPTRec, SIGIR workshop 2023, [[paper]](https://arxiv.org/abs/2306.11114.pdf)
- FANS: Fast Non-Autoregressive Sequence Generation for Item List Continuation, WWW 2023, [[paper]](https://arxiv.org/pdf/2304.00545.pdf)
- Generative Next-Basket Recommendation, RecSys 2023
- Large Language Model Augmented Narrative Driven Recommendations, RecSys 2023, [[paper]](https://arxiv.org/pdf/2304.00545.pdf)


## Dataset
- PeterRec dataset[[link]](https://drive.google.com/file/d/1OcvbBJN0jlPTEjE0lvcDfXRkzOjepMXH/view)
- Tenrec: A Large-scale Multipurpose Benchmark Dataset for Recommender Systems, NeurIPS 2022  [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/4ad4fc1528374422dd7a69dea9e72948-Paper-Datasets_and_Benchmarks.pdf)
- Where to Go Next for Recommender Systems? ID-vs. Modality-based recommender models revisited, openreview 2022/09, [[paper]](https://arxiv.org/pdf/2303.13835.pdf)











## Reference:
- Xiangyang Li Github  [[Link]](https://github.com/archersama/awesome-recommend-system-pretraining-papers/tree/main)
- nancheng58i Github  [[Link]](https://github.com/nancheng58)


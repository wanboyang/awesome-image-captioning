# Awesome Image Captioning[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<p align="center">
  <img width="250" src="https://camo.githubusercontent.com/1131548cf666e1150ebd2a52f44776d539f06324/68747470733a2f2f63646e2e7261776769742e636f6d2f73696e647265736f726875732f617765736f6d652f6d61737465722f6d656469612f6c6f676f2e737667" "Awesome!">
</p>

A curated list of image captioning and related area. :-)

## Contributing
Please feel free to send me [pull requests](https://github.com/zhjohnchan/awesome-image-captioning/pulls) or email (chihung.chan@outlook.com) to add links.
Markdown format:
```markdown
- [Paper Name](link) - Author 1 et al, `Conference Year`. [[code]](link)
```
## Change Log
- Mar. 06 [Another related repository](https://github.com/zhjohnchan/awesome-radiology-report-generation) in medical domain updated
- Jun. 13 CVPR'20 and ACL'20 updated

## Table of Contents
- [Papers](#papers)
  - [Survey](#survey)
  - [Before](#before) - [2015](#2015) - [2016](#2016) - [2017](#2017) - [2018](#2018) - [2019](#2019)
- [Dataset](#dataset)
- [Image Captioning Challenge](#image-captioning-challenge)
- [Popular Implementations](#popular-implementations)
  - [PyTorch](#pytorch)
  - [TensorFlow](#tensorflow)
  - [Torch](#torch)
  - [Others](#others)

## Papers
### Survey
* [A Comprehensive Survey of Deep Learning for Image Captioning](https://arxiv.org/abs/1810.04020) - Hossain M et al, `arXiv preprint 2018`.
### Before
* [I2t: Image parsing to text description](https://ieeexplore.ieee.org/abstract/document/5487377/) - Yao B Z et al, `P IEEE 2011`.
* [Im2Text: Describing Images Using 1 Million Captioned Photographs](http://tamaraberg.com/papers/generation_nips2011.pdf) - Ordonez V et al, `NIPS 2011`. [[project web]](http://vision.cs.stonybrook.edu/~vicente/sbucaptions/)
* [Deep Captioning with Multimodal Recurrent Neural Networks](http://arxiv.org/abs/1412.6632) - Mao J et al, `arXiv preprint 2014`.

### 2015
#### *`CVPR 2015`*
* [Show and Tell: A Neural Image Caption Generator](http://arxiv.org/abs/1411.4555) - Vinyals O et al, `CVPR 2015`. [[code]](https://github.com/karpathy/neuraltalk) [[code]](https://github.com/zsdonghao/Image-Captioning)
* [Deep Visual-Semantic Alignments for Generating Image Descriptions](http://arxiv.org/abs/1412.2306) - Karpathy A et al, `CVPR 2015`. [[project web]](http://cs.stanford.edu/people/karpathy/deepimagesent/) [[code]](https://github.com/karpathy/neuraltalk)
* [Mind’s Eye: A Recurrent Visual Representation for Image Caption Generation](http://www.cs.cmu.edu/~xinleic/papers/cvpr15_rnn.pdf) - Chen X et al, `CVPR 2015`.
* [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](http://arxiv.org/abs/1411.4389) - Donahue J et al, `CVPR 2015`. [[code]](https://github.com/BVLC/caffe/pull/2033) [[project web]](http://jeffdonahue.com/lrcn/)
#### *`ICCV 2015`*
* [Guiding the Long-Short Term Memory Model for Image Caption Generation](https://arxiv.org/abs/1509.04942) - Jia X et al, `ICCV 2015`.
* [Learning like a Child: Fast Novel Visual Concept Learning from Sentence Descriptions of Images](http://arxiv.org/abs/1504.06692) - Mao J et al, `ICCV 2015`. [[code]](https://github.com/mjhucla/NVC-Dataset)
#### *`NIPS 2015`*
* [Expressing an Image Stream with a Sequence of Natural Sentences](http://papers.nips.cc/paper/5776-expressing-an-image-stream-with-a-sequence-of-natural-sentences.pdf) - Park C C et al, `NIPS 2015`. [[code]](https://github.com/cesc-park/CRCN)
#### *`ICML 2015`*
* [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://arxiv.org/abs/1502.03044) - Xu K et al, `ICML 2015`. [[project]](http://kelvinxu.github.io/projects/capgen.html) [[code]](https://github.com/yunjey/show-attend-and-tell-tensorflow) [[code]](https://github.com/kelvinxu/arctic-captions)
#### *`arXiv preprint 2015`*
* [Order-Embeddings of Images and Language](http://arxiv.org/abs/1511.06361) - Vendrov I et al, `arXiv preprint 2015`. [[code]](https://github.com/ivendrov/order-embedding)
* [Generating Images from Captions with Attention](http://arxiv.org/abs/1511.02793) - Mansimov E et al, `arXiv preprint 2015`. [[code]](https://github.com/emansim/text2image)
* [Learning FRAME Models Using CNN Filters for Knowledge Visualization](http://arxiv.org/abs/1509.08379) - Lu Y, et al, `arXiv preprint 2015`. [[code]](http://www.stat.ucla.edu/~yang.lu/project/deepFrame/doc/deepFRAME_1.1.zip) 
* [Aligning where to see and what to tell: image caption with region-based attention and scene factorization](http://arxiv.org/abs/1506.06272) - Jin J et al, `arXiv preprint 2015`.

### 2016
#### *`CVPR 2016`*
* [Image captioning with semantic attention](https://arxiv.org/abs/1603.03925) - You Q et al, `CVPR 2016`. [[code]](https://github.com/chapternewscu/image-captioning-with-semantic-attention)
* [DenseCap: Fully Convolutional Localization Networks for Dense Captioning](http://arxiv.org/abs/1511.07571) - Johnson J et al, `CVPR 2016`. [[code]](https://github.com/jcjohnson/densecap)
* [What value do explicit high level concepts have in vision to language problems?](http://arxiv.org/abs/1506.01144) - Wu Q et al, `CVPR 2016`.
* [Deep Compositional Captioning: Describing Novel Object Categories without Paired Training Data](https://arxiv.org/abs/1511.05284#) - Lisa Anne Hendricks et al, `CVPR 2016`. [[code]](https://github.com/LisaAnne/DCC)
* [SPICE: Semantic Propositional Image Caption Evaluation](http://www.panderson.me/images/SPICE.pdf) - Anderson P et al, `ECCV 2016`. [[code]](https://github.com/peteanderson80/SPICE)
#### *`ACMMM 2016`*
* [Image Captioning with Deep Bidirectional LSTMs](http://arxiv.org/abs/1604.00790) - Wang C et al, `ACMMM 2016`. [[code]](https://github.com/deepsemantic/image_captioning)
#### *`ACL 2016`*
* [Multimodal Pivots for Image Caption Translation](http://arxiv.org/abs/1511.02793) - Hitschler J et al, `ACL 2016`.
#### *`arXiv preprint 2016`*
* [Image Caption Generation with Text-Conditional Semantic Attention](https://arxiv.org/abs/1606.04621) - Zhou L et al, `arXiv preprint 2016`. [[code]](https://github.com/LuoweiZhou/e2e-gLSTM-sc)
* [DeepDiary: Automatic Caption Generation for Lifelogging Image Streams](http://arxiv.org/abs/1608.03819) - Fan C et al, `arXiv preprint 2016`.
* [Learning to generalize to new compositions in image understanding](http://arxiv.org/abs/1608.07639) - Atzmon Y et al, `arXiv preprint 2016`.
* [Generating captions without looking beyond objects](https://arxiv.org/abs/1610.03708) - Heuer H et al, `arXiv preprint 2016`.
* [Bootstrap, Review, Decode: Using Out-of-Domain Textual Data to Improve Image Captioning](https://arxiv.org/abs/1611.05321) - Chen W et al, `arXiv preprint 2016`. [[code]](https://github.com/wenhuchen/Semi-Supervised-Image-Captioning)
* [Recurrent Image Captioner: Describing Images with Spatial-Invariant Transformation and Attention Filtering](https://arxiv.org/abs/1612.04949) - Liu H et al, `arXiv preprint 2016`.
* [Recurrent Highway Networks with Language CNN for Image Captioning](https://arxiv.org/abs/1612.07086) - Gu J et al, `arXiv preprint  2016`.

### 2017
#### *`CVPR 2017`*
* [Captioning Images with Diverse Objects](http://arxiv.org/abs/1606.07770) - Venugopalan S et al, `CVPR 2017`. [[code]](https://github.com/vsubhashini/noc)
* [Top-down Visual Saliency Guided by Captions](https://arxiv.org/abs/1612.07360) - Ramanishka V et al, `CVPR 2017`. [[code]](https://github.com/VisionLearningGroup/caption-guided-saliency)
* [Self-Critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563) - Steven J et al, `CVPR 2017`. [[code]](https://github.com/ruotianluo/self-critical.pytorch)
* [Dense Captioning with Joint Inference and Visual Context](https://arxiv.org/abs/1611.06949) - Yang L et al, `CVPR 2017`. [[code]](https://github.com/linjieyangsc/densecap)
* [Skeleton Key: Image Captioning by Skeleton-Attribute Decomposition](https://arxiv.org/abs/1704.06972) - Yufei W et al, `CVPR 2017`. [[code]](https://github.com/feiyu1990/Skeleton-key)
* [A Hierarchical Approach for Generating Descriptive Image Paragraphs](https://arxiv.org/abs/1611.06607) - Krause J et al, `CVPR 2017`. [[code]](https://github.com/InnerPeace-Wu/im2p-tensorflow)
* [Deep Reinforcement Learning-based Image Captioning with Embedding Reward](https://arxiv.org/abs/1704.03899) - Ren Z et al, `CVPR 2017`.
* [Incorporating Copying Mechanism in Image Captioning for Learning Novel Objects](https://arxiv.org/abs/1708.05271) - Ting Y et al, `CVPR 2017`.
* [Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning](https://arxiv.org/abs/1612.01887) - Lu J et al, `CVPR 2017`. [[code]](https://github.com/jiasenlu/AdaptiveAttention)
* [Attend to You: Personalized Image Captioning with Context Sequence Memory Networks](https://arxiv.org/abs/1704.06485) - CC Park et al, `CVPR 2017`. [[code]](https://github.com/cesc-park/attend2u)
* [SCA-CNN: Spatial and channel-wise attention in convolutional networks for image captioning](https://arxiv.org/abs/1611.05594) - Chen L et al, `CVPR 2017`. [[code]](https://github.com/zjuchenlong/sca-cnn.cvpr17)
* [Bidirectional Beam Search: Forward-Backward Inference in Neural Sequence Models for Fill-In-The-Blank Image Captioning](https://arxiv.org/abs/1705.08759) - Qing S et al, `CVPR 2017`.
#### *`ICCV 2017`*
* [Areas of Attention for Image Captioning](https://arxiv.org/abs/1612.01033) - Pedersoli M et al, `ICCV 2017`.
* [Boosting Image Captioning with Attributes](https://arxiv.org/abs/1611.01646) - Yao T et al, `ICCV 2017`.
* [An Empirical Study of Language CNN for Image Captioning](https://arxiv.org/abs/1612.07086) - Gu J et al, `ICCV 2017`.
* [Improved Image Captioning via Policy Gradient Optimization of SPIDEr](https://arxiv.org/abs/1612.00370) - Liu S et al, `ICCV 2017`.
* [Towards Diverse and Natural Image Descriptions via a Conditional GAN](http://openaccess.thecvf.com/content_ICCV_2017/papers/Dai_Towards_Diverse_and_ICCV_2017_paper.pdf) - Dai B et al, `ICCV 2017`. [[code]](https://github.com/doubledaibo/gancaption_iccv2017)
* [Paying Attention to Descriptions Generated by Image Captioning Models](https://arxiv.org/abs/1704.07434) - Tavakoliy H R et al, `ICCV 2017`.
* [Show, Adapt and Tell: Adversarial Training of Cross-domain Image Captioner](https://arxiv.org/abs/1705.00930) - Chen T H et al, `ICCV 2017`. [[code]](https://github.com/tsenghungchen/show-adapt-and-tell)
#### *`AAAI 2017`*
* [Image Caption with Global-Local Attention](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14880/14291) - Li L et al, `AAAI 2017`.
* [Reference Based LSTM for Image Captioning](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14249/14270) - Chen M et al, `AAAI 2017`.
* [Attention Correctness in Neural Image Captioning](https://arxiv.org/abs/1605.09553) - Liu C et al, `AAAI 2017`.
* [Text-guided Attention Model for Image Captioning](https://arxiv.org/abs/1612.03557) - Mun J et al, `AAAI 2017`. [[code]](https://github.com/JonghwanMun/TextguidedATT)
#### *`NIPS 2017`*
* [Contrastive Learning for Image Captioning](https://arxiv.org/abs/1710.02534) - Dai B et al, `NIPS 2017`. [[code]](https://github.com/doubledaibo/clcaption_nips2017)
#### *`TPAMI 2017`*
* [Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge](http://arxiv.org/abs/1609.06647) - Vinyals O et al, `TPAMI 2017`. [[code]](https://github.com/tensorflow/models/tree/master/im2txt)
#### *`arXiv preprint 2017`*
* [MAT: A Multimodal Attentive Translator for Image Captioning](https://arxiv.org/abs/1702.05658) - Liu C et al, `arXiv preprint  2017`.
* [Actor-Critic Sequence Training for Image Captioning](https://arxiv.org/abs/1706.09601) - Zhang L et al, `arXiv preprint 2017`.
* [What is the Role of Recurrent Neural Networks (RNNs) in an Image Caption Generator?](https://arxiv.org/abs/1708.02043) - Tanti M et al, `arXiv preprint 2017`. [[code]](https://github.com/mtanti/rnn-role)
* [Self-Guiding Multimodal LSTM - when we do not have a perfect training dataset for image captioning](https://arxiv.org/abs/1709.05038) - Xian Y et al, `arXiv preprint 2017`.
* [Phrase-based Image Captioning with Hierarchical LSTM Model](https://arxiv.org/abs/1711.05557) - Tan Y H et al, `arXiv preprint 2017`.
* [Show-and-Fool: Crafting Adversarial Examples for Neural Image Captioning](https://arxiv.org/abs/1712.02051) - Chen H et al, `arXiv preprint 2017`.

### 2018
#### *`CVPR 2018`*
* [Neural Baby Talk](https://arxiv.org/abs/1803.09845) - Lu J et al, `CVPR 2018`. [[code]](https://github.com/jiasenlu/NeuralBabyTalk)
* [Convolutional Image Captioning](https://arxiv.org/abs/1711.09151) - Aneja J et al, `CVPR 2018`. [[code]](https://github.com/aditya12agd5/convcap)  
* [Learning to Evaluate Image Captioning](https://arxiv.org/abs/1806.06422) - Cui Y et al, `CVPR 2018`. [[code]](https://github.com/richardaecn/cvpr18-caption-eval)
* [Discriminability Objective for Training Descriptive Captions](https://arxiv.org/abs/1803.04376) - Luo R et al, `CVPR 2018`. [[code]](https://github.com/ruotianluo/DiscCaptioning)
* [SemStyle: Learning to Generate Stylised Image Captions using Unaligned Text](https://arxiv.org/abs/1805.07030) - Mathews A et al, `CVPR 2018`.[[code]](https://github.com/computationalmedia/semstyle)  
* [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998) - Anderson P et al, `CVPR 2018`. [[code]](https://github.com/peteanderson80/bottom-up-attention)
* [GroupCap: Group-Based Image Captioning With Structured Relevance and Diversity Constraints
](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_GroupCap_Group-Based_Image_CVPR_2018_paper.pdf) - Chen F et al, `CVPR 2018`.
#### *`ECCV 2018`*
* [Unpaired Image Captioning by Language Pivoting](http://openaccess.thecvf.com/content_ECCV_2018/papers/Jiuxiang_Gu_Unpaired_Image_Captioning_ECCV_2018_paper.pdf) - Gu J et al, `ECCV 2018`.
* [Recurrent Fusion Network for Image Captioning](https://arxiv.org/abs/1807.09986) - Jiang W et al, `ECCV 2018`.
* [Exploring Visual Relationship for Image Captioning](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ting_Yao_Exploring_Visual_Relationship_ECCV_2018_paper.pdf) - Yao T et al, `ECCV 2018`.
* [Rethinking the Form of Latent States in Image Captioning](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bo_Dai_Rethinking_the_Form_ECCV_2018_paper.pdf) - Dai B et al, `ECCV 2018`. [[code]](https://github.com/doubledaibo/2dcaption_eccv2018)
* [Boosted Attention: Leveraging Human Attention for Image Captioning](http://openaccess.thecvf.com/content_ECCV_2018/papers/Shi_Chen_Boosted_Attention_Leveraging_ECCV_2018_paper.pdf) - Chen S et al, `ECCV 2018`.
* ["Factual" or "Emotional": Stylized Image Captioning with Adaptive Learning and Attention](https://arxiv.org/abs/1807.03871) - Chen T et al, `ECCV 2018`.
#### *`AAAI 2018`*
* [Learning to Guide Decoding for Image Captioning](https://arxiv.org/abs/1804.00887) - Jiang W et al, `AAAI 2018`.[[code]](https://github.com/HaleyPei/Learning-to-Guide-Decoding-for-Image-Captioning)  
* [Stack-Captioning: Coarse-to-Fine Learning for Image Captioning](https://arxiv.org/abs/1709.03376) - Gu J et al, `AAAI 2018`. [[code]](https://github.com/gujiuxiang/Stack-Captioning)
* [Temporal-difference Learning with Sampling Baseline for Image Captioning](http://eprints.lancs.ac.uk/123576/1/2018_4.pdf) - Chen H et al, `AAAI 2018`.
#### *`NeurIPS 2018`*
* [Partially-Supervised Image Captioning](https://arxiv.org/pdf/1806.06004.pdf) - Anderson P et al, `NeurIPS 2018`.[[code]](https://github.com/ajaysub110/A-Neural-Compositional-Paradigm-for-Image-Captioning)  
* [A Neural Compositional Paradigm for Image Captioning](https://arxiv.org/pdf/1810.09630.pdf) - Dai B et al, `NeurIPS 2018`.
#### *`NAACL 2018`*
* [Defoiling Foiled Image Captions](https://arxiv.org/abs/1805.06549) - Wang J et al, `NAACL 2018`.
* [Punny Captions: Witty Wordplay in Image Descriptions](https://arxiv.org/abs/1704.08224) - Chandrasekaran A et al, `NAACL 2018`. [[code]](https://github.com/purvaten/punny_captions)
* [Object Counts! Bringing Explicit Detections Back into Image Captioning](https://arxiv.org/abs/1805.00314) - Aneja J et al, `NAACL 2018`.
#### *`ACL 2018`*
* [Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning](http://www.aclweb.org/anthology/P18-1238) - Sharma P et al, `ACL 2018`. [[code]](https://github.com/google-research-datasets/conceptual-captions)
* [Attacking visual language grounding with adversarial examples: A case study on neural image captioning](http://www.aclweb.org/anthology/P18-1241) - Chen H et al, `ACL 2018`. [[code]](https://github.com/IBM/Image-Captioning-Attack)
#### *`EMNLP 2018`*
* [simNet: Stepwise Image-Topic Merging Network for Generating Detailed and Comprehensive Image Captions](http://aclweb.org/anthology/D18-1013.pdf) - Liu et al, `EMNLP 2018`. [[code]](https://github.com/lancopku/simNet)
#### *`arXiv preprint 2018`*
* [Improved Image Captioning with Adversarial Semantic Alignment](https://arxiv.org/abs/1805.00063) - Melnyk I et al, `arXiv preprint 2018`.
* [Improving Image Captioning with Conditional Generative Adversarial Nets](https://arxiv.org/abs/1805.07112) - Chen C et al, `arXiv preprint 2018`.
* [CNN+CNN: Convolutional Decoders for Image Captioning](https://arxiv.org/abs/1805.09019) - Wang Q et al, `arXiv preprint 2018`.[[code]](https://github.com/qingzwang/GHA-ImageCaptioning)
* [Diverse and Controllable Image Captioning with Part-of-Speech Guidance](https://arxiv.org/abs/1805.12589) - Deshpande A et al, `arXiv preprint 2018`.

### 2019
#### *`CVPR 2019`*
* [Unsupervised Image Captioning](https://arxiv.org/abs/1811.10787) - Yang F et al, `CVPR 2019`. [[code]](https://github.com/fengyang0317/unsupervised_captioning)
* [Engaging Image Captioning Via Personality](https://arxiv.org/abs/1810.10665) - Shuster K et al, `CVPR 2019`.
* [Pointing Novel Objects in Image Captioning](https://arxiv.org/abs/1904.11251) - Li Y et al, `CVPR 2019`.
* [Auto-Encoding Scene Graphs for Image Captioning](https://arxiv.org/abs/1812.02378) - Yang X et al, `CVPR 2019`.[[code]](https://github.com/fengyang0317/unsupervised_captioning)   
* [Context and Attribute Grounded Dense Captioning](https://arxiv.org/abs/1904.01410) - Yin G et al, `CVPR 2019`.
* [Look Back and Predict Forward in Image Captioning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Qin_Look_Back_and_Predict_Forward_in_Image_Captioning_CVPR_2019_paper.pdf) - Qin Y et al, `CVPR 2019`.
* [Self-critical n-step Training for Image Captioning](https://arxiv.org/abs/1904.06861) - Gao J et al, `CVPR 2019`.
* [Intention Oriented Image Captions with Guiding Objects](https://arxiv.org/abs/1811.07662) - Zheng Y et al, `CVPR 2019`.
* [Describing like humans: on diversity in image captioning](https://arxiv.org/abs/1903.12020) - Wang Q et al, `CVPR 2019`.[[code]](https://github.com/qingzwang/DiversityMetrics)  
* [Adversarial Semantic Alignment for Improved Image Captions](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dognin_Adversarial_Semantic_Alignment_for_Improved_Image_Captions_CVPR_2019_paper.pdf) - Dognin P et al, `CVPR 2019`.
* [MSCap: Multi-Style Image Captioning With Unpaired Stylized Text](http://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_MSCap_Multi-Style_Image_Captioning_With_Unpaired_Stylized_Text_CVPR_2019_paper.pdf) - Gao L et al, `CVPR 2019`.
* [Fast, Diverse and Accurate Image Captioning Guided By Part-of-Speech](https://arxiv.org/abs/1805.12589) - Aditya D et al, `CVPR 2019`.
* [Good News, Everyone! Context driven entity-aware captioning for news images](https://arxiv.org/abs/1904.01475) - Biten A F et al, `CVPR 2019`. [[code]](https://github.com/furkanbiten/GoodNews)
* [CapSal: Leveraging Captioning to Boost Semantics for Salient Object Detection](https://github.com/zhangludl/code-and-dataset-for-CapSal) - Zhang L et al, `CVPR 2019`. [[code]](https://github.com/zhangludl/code-and-dataset-for-CapSal)
* [Dense Relational Captioning: Triple-Stream Networks for Relationship-Based Captioning](https://arxiv.org/abs/1903.05942) - Kim D et al, `CVPR 2019`. [[code]](https://github.com/Dong-JinKim/DenseRelationalCaptioning)
* [Show, Control and Tell: A Framework for Generating Controllable and Grounded Captions](https://arxiv.org/abs/1811.10652v2) - Cornia M et al, `CVPR 2019`. [[code]](https://github.com/aimagelab/show-control-and-tell)  
* [Exact Adversarial Attack to Image Captioning via Structured Output Learning With Latent Variables](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Exact_Adversarial_Attack_to_Image_Captioning_via_Structured_Output_Learning_CVPR_2019_paper.pdf) - Xu Y et al, `CVPR 2019`.
#### *`AAAI 2019`*
* [Meta Learning for Image Captioning](https://aaai.org/ojs/index.php/AAAI/article/view/4883/4756) - Li N et al, `AAAI 2019`.
* [Learning Object Context for Dense Captioning](http://vipl.ict.ac.cn/homepage/jsq/publication/2018-Li-AAAI-Learning-Object-Context-for-Dense-Captioning.pdf) - Li X et al, `AAAI 2019`.
* [Hierarchical Attention Network for Image Captioning](https://aaai.org/ojs/index.php/AAAI/article/view/4924/4797) - Wang W et al, `AAAI 2019`.
* [Deliberate Residual based Attention Network for Image Captioning](https://www.aaai.org/Papers/AAAI/2019/AAAI-GaoLianli3.5390.pdf) - Gao L et al, `AAAI 2019`.
* [Improving Image Captioning with Conditional Generative Adversarial Nets](https://arxiv.org/abs/1805.07112) - Chen C et al, `AAAI 2019`.
* [Connecting Language to Images: A Progressive Attention-Guided Network for Simultaneous Image Captioning and Language Grounding](https://aaai.org/ojs/index.php/AAAI/article/view/4916/4789) - Song L et al, `AAAI 2019`.
#### *`ACL 2019`*
* [Dense Procedure Captioning in Narrated Instructional Videos](https://www.msra.cn/wp-content/uploads/2019/06/DenseProcedureCaptioninginNarratedInstructionalVideos.pdf) - Shi B et al, `ACL 2019`.
* [Informative Image Captioning with External Sources of Information](https://arxiv.org/pdf/1906.08876.pdf) - Zhao S et al, `ACL 2019`.
* [Bridging by Word: Image Grounded Vocabulary Construction for Visual Captioning](https://www.aclweb.org/anthology/P19-1652) - Fan Z et al, `ACL 2019`.
#### *`BMVC 2019`*
* [Image Captioning with Unseen Objects](https://bmvc2019.org/wp-content/uploads/papers/0124-paper.pdf) - Demirel et al, `BMVC 2019`.
* [Look and Modify: Modification Networks for Image Captioning](https://arxiv.org/abs/1909.03169) - Sammani et al, `BMVC 2019`. [[code]](https://github.com/fawazsammani/look-and-modify)
* [Show, Infer and Tell: Contextual Inference for Creative Captioning](https://bmvc2019.org/wp-content/uploads/papers/0655-paper.pdf) - Khare et al, `BMVC 2019`. [[code]](https://github.com/ankit1khare/Show_Infer_and_Tell-CIC)
* [SC-RANK: Improving Convolutional Image Captioning with Self-Critical Learning and Ranking Metric-based Reward](https://bmvc2019.org/wp-content/uploads/papers/0936-paper.pdf) - Yan et al, `BMVC 2019`.
#### *`ICCV 2019`*
- **DUDA**: Robust Change Captioning  
  *Dong Huk Park, Trevor Darrell, Anna Rohrbach*  [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Robust_Change_Captioning_ICCV_2019_paper.pdf)    
  `ICCV 2019 Oral`

- **AoANet**: Attention on Attention for Image Captioning  [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_Attention_on_Attention_for_Image_Captioning_ICCV_2019_paper.pdf)
[[code]](https://github.com/husthuaan/AoANet)  
  *Lun Huang, Wenmin Wang, Jie Chen, Xiao-Yong Wei*   
  `ICCV 2019 Oral`

- **MaBi-LSTMs**: Exploring Overall Contextual Information for Image Captioning in Human-Like Cognitive Style  [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ge_Exploring_Overall_Contextual_Information_for_Image_Captioning_in_Human-Like_Cognitive_ICCV_2019_paper.pdf)   
  *Hongwei Ge, Zehang Yan, Kai Zhang, Mingde Zhao, Liang Sun*  

- **Align2Ground**: Align2Ground: Weakly Supervised Phrase Grounding Guided by Image-Caption Alignment  [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Datta_Align2Ground_Weakly_Supervised_Phrase_Grounding_Guided_by_Image-Caption_Alignment_ICCV_2019_paper.pdf)  
  Samyak Datta, Karan Sikka, Anirban Roy, Karuna Ahuja, Devi Parikh, Ajay Divakaran*    

- **GCN-LSTM+HIP**: Hierarchy Parsing for Image Captioning [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yao_Hierarchy_Parsing_for_Image_Captioning_ICCV_2019_paper.pdf)  
  *Ting Yao, Yingwei Pan, Yehao Li, Tao Mei*    

- **IR+Tdiv**: Generating Diverse and Descriptive Image Captions Using Visual Paraphrases   [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Generating_Diverse_and_Descriptive_Image_Captions_Using_Visual_Paraphrases_ICCV_2019_paper.pdf)  
  *Lixin Liu, Jiajun Tang, Xiaojun Wan, Zongming Guo*    

- **CNM+SGAE**: Learning to Collocate Neural Modules for Image Captioning  [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_Learning_to_Collocate_Neural_Modules_for_Image_Captioning_ICCV_2019_paper.pdf)  
  *Xu Yang, Hanwang Zhang, Jianfei Cai*     

- **Seq-CVAE**: Sequential Latent Spaces for Modeling the Intention During Diverse Image Captioning  [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Aneja_Sequential_Latent_Spaces_for_Modeling_the_Intention_During_Diverse_Image_ICCV_2019_paper.pdf)   
  Jyoti Aneja, Harsh Agrawal, Dhruv Batra, Alexander Schwing

- Towards Unsupervised Image Captioning With Shared Multimodal Embeddings  [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Laina_Towards_Unsupervised_Image_Captioning_With_Shared_Multimodal_Embeddings_ICCV_2019_paper.pdf)   
  *Iro Laina, Christian Rupprecht, Nassir Navab*    

- Human Attention in Image Captioning: Dataset and Analysis  [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/He_Human_Attention_in_Image_Captioning_Dataset_and_Analysis_ICCV_2019_paper.pdf) [[code]](https://github.com/SenHe/Human-Attention-in-Image-Captioning)   
  *Sen He, Hamed R. Tavakoli, Ali Borji, Nicolas Pugeault*   

- **RDN**: Reflective Decoding Network for Image Captioning  [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ke_Reflective_Decoding_Network_for_Image_Captioning_ICCV_2019_paper.pdf)  
  *Lei Ke, Wenjie Pei, Ruiyu Li, Xiaoyong Shen, Yu-Wing Tai*   

- **PSST**: Joint Optimization for Cooperative Image Captioning  [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Vered_Joint_Optimization_for_Cooperative_Image_Captioning_ICCV_2019_paper.pdf)   
  *Gilad Vered, Gal Oren, Yuval Atzmon, Gal Chechik*   

- **MUTAN**: Watch, Listen and Tell: Multi-Modal Weakly Supervised Dense Event Captioning  [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Rahman_Watch_Listen_and_Tell_Multi-Modal_Weakly_Supervised_Dense_Event_Captioning_ICCV_2019_paper.pdf)  
  *Tanzila Rahman, Bicheng Xu, Leonid Sigal*   

- **ETA**: Entangled Transformer for Image Captioning   [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Entangled_Transformer_for_Image_Captioning_ICCV_2019_paper.pdf)  
  *Guang Li, Linchao Zhu, Ping Liu, Yi Yang*   

- **nocaps**: novel object captioning at scale  [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Agrawal_nocaps_novel_object_captioning_at_scale_ICCV_2019_paper.pdf)  
  *Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, Peter Anderson*   

- **Cap2Det**: Learning to Amplify Weak Caption Supervision for Object Detection  [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ye_Cap2Det_Learning_to_Amplify_Weak_Caption_Supervision_for_Object_Detection_ICCV_2019_paper.pdf) 
[[code]](https://github.com/yekeren/Cap2Det)   
  *Keren Ye, Mingda Zhang, Adriana Kovashka, Wei Li, Danfeng Qin, Jesse Berent*   

- **Graph-Align**: Unpaired Image Captioning via Scene Graph Alignments  [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gu_Unpaired_Image_Captioning_via_Scene_Graph_Alignments_ICCV_2019_paper.pdf)  
  *Jiuxiang* Gu, Shafiq Joty, Jianfei Cai, Handong Zhao, Xu Yang, Gang Wang   

- : Learning to Caption Images Through a Lifetime by Asking Questions  [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shen_Learning_to_Caption_Images_Through_a_Lifetime_by_Asking_Questions_ICCV_2019_paper.pdf)  
  *Tingke Shen, Amlan Kar, Sanja Fidler*   
#### *`NeurIPS 2019`*
* [Image Captioning: Transforming Objects into Words](https://papers.nips.cc/paper/9293-image-captioning-transforming-objects-into-words) - Herdade S et al, `NeurIPS 2019`. [[code]](https://github.com/yahoo/object_relation_transformer)  
* [Adaptively Aligned Image Captioning via Adaptive Attention Time](https://arxiv.org/pdf/1909.09060.pdf) - Huang L et al, `NeurIPS 2019`. [[code]](https://github.com/husthuaan/AAT)
* [Variational Structured Semantic Inference for Diverse Image Captioning](https://papers.nips.cc/paper/8468-variational-structured-semantic-inference-for-diverse-image-captioning) - Chen F et al, `NeurIPS 2019`.
* [Aligning Visual Regions and Textual Concepts for Semantic-Grounded Image Representations](https://papers.nips.cc/paper/8909-aligning-visual-regions-and-textual-concepts-for-semantic-grounded-image-representations.pdf) - Liu F et al, `NeurIPS 2019`. [[code]](https://github.com/fenglinliu98/MIA)
#### *`IJCAI 2019`*
* [Image Captioning with Compositional Neural Module Networks](https://www.researchgate.net/profile/Junjiao_Tian/publication/334844176_Image_Captioning_with_Compositional_Neural_Module_Networks/links/5d784f49a6fdcc9961bfb7fd/Image-Captioning-with-Compositional-Neural-Module-Networks.pdf) - Tian J et al, `IJCAI 2019`.
* [Exploring and Distilling Cross-Modal Information for Image Captioning](https://www.ijcai.org/proceedings/2019/708) - Liu F et al, `IJCAI 2019`.
* [Swell-and-Shrink: Decomposing Image Captioning by Transformation and Summarization](https://www.ijcai.org/proceedings/2019/0726.pdf) - Wang H et al, `IJCAI 2019`.
* [Hornet: a hierarchical offshoot recurrent network for improving person re-ID via image captioning](https://arxiv.org/pdf/1908.04915) - Yan S et al, `IJCAI 2019`.
#### *`EMNLP 2019`*
* [Image Captioning with Very Scarce Supervised Data: Adversarial Semi-Supervised Learning Approach](https://arxiv.org/pdf/1909.02201) - Kim D J et al, `EMNLP 2019`.
* [TIGEr: Text-to-Image Grounding for Image Caption Evaluation](https://arxiv.org/pdf/1909.02050) - Jiang M et al, `EMNLP 2019`.
* [REO-Relevance, Extraness, Omission: A Fine-grained Evaluation for Image Captioning](https://arxiv.org/pdf/1909.02217) - Jiang M et al, `EMNLP 2019`.
* [Decoupled Box Proposal and Featurization with Ultrafine-Grained Semantic Labels Improve Image Captioning and Visual Question Answering](https://arxiv.org/pdf/1909.02097) - Changpinyo S et al, `EMNLP 2019`.

### 2020
#### *`AAAI 2020`*
- **Unified VLP**: Unified Vision-Language Pre-Training for Image Captioning and VQA  [[paper]](https://arxiv.org/abs/1909.11059)
[[code]](https://github.com/LuoweiZhou/VLP)  
  *Luowei Zhou (University of Michigan); Hamid Palangi (Microsoft Research); Lei Zhang (Microsoft); Houdong Hu  (Microsoft AI and Research); Jason Corso (University of Michigan); Jianfeng Gao (Microsoft Research)*  

- **OffPG**: Reinforcing an Image Caption Generator using Off-line Human Feedback  [[paper]](https://arxiv.org/abs/1911.09753)  
  *Paul Hongsuck Seo (POSTECH); Piyush Sharma (Google Research); Tomer Levinboim (Google); Bohyung Han(Seoul National University); Radu Soricut (Google)*  

- **MemCap**: Memorizing Style Knowledge for Image Captioning  [[paper]](https://wuxinxiao.github.io/assets/papers/2020/MemCap.pdf)  
  *Wentian Zhao (Beijing Institute of Technology); Xinxiao Wu (Beijing Institute of Technology); Xiaoxun Zhang(Alibaba Group)*  

- **C-R Reasoning**: Joint Commonsense and Relation Reasoning for Image and Video Captioning  [[paper]](https://wuxinxiao.github.io/assets/papers/2020/C-R_reasoning.pdf)  
  *Jingyi Hou (Beijing Institute of Technology); Xinxiao Wu (Beijing Institute of Technology); Xiaoxun Zhang (AlibabaGroup); Yayun Qi (Beijing Institute of Technology); Yunde Jia (Beijing Institute of Technology); Jiebo Luo (University of Rochester)*

- **MHTN**: Learning Long- and Short-Term User Literal-Preference with Multimodal Hierarchical Transformer Network for Personalized Image Caption  [[paper]](https://weizhangltt.github.io/paper/zhang-aaai20.pdf)  
  *Wei Zhang (East China Normal University); Yue Ying (East China Normal University); Pan Lu (The University of California, Los Angeles); Hongyuan Zha (GEORGIA TECH)*  

- Show, Recall, and Tell: Image Captioning with Recall Mechanism [[paper]](https://arxiv.org/abs/2001.05876)  
  *Li WANG (MoE Key Lab of Artificial Intelligence, AI Institute, Shanghai Jiao Tong University, China); Zechen BAI(Institute of Software, Chinese Academy of Science, China); Yonghua Zhang (Bytedance); Hongtao Lu (Shanghai Jiao Tong University)*   

- Interactive Dual Generative Adversarial Networks for Image Captioning  [[paper]](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-LiuJ.3970.pdf)  
  *Junhao Liu (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences); Kai Wang (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences); Chunpu Xu (Huazhong University of Science and Technology); Zhou Zhao (Zhejiang University); Ruifeng Xu (Harbin Institute of Technology (Shenzhen)); Ying Shen (Peking University Shenzhen Graduate School); Min Yang ( Chinese Academy of Sciences)*  

- **FDM-net**: Feature Deformation Meta-Networks in Image Captioning of Novel Objects [[paper]](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-CaoT.4566.pdf)  
  Tingjia Cao (Fudan University); Ke Han (Fudan University); Xiaomei Wang (Fudan University); Lin Ma (Tencent AI Lab); Yanwei Fu (Fudan University); Yu-Gang Jiang (Fudan University); Xiangyang Xue (Fudan University)

#### *`CVPR 2020`*
- Context-Aware Group Captioning via Self-Attention and Contrastive Features [[paper]](https://arxiv.org/abs/2004.03708)  
  Zhuowan Li, Quan Tran, Long Mai, Zhe Lin, Alan L. Yuille
- More Grounded Image Captioning by Distilling Image-Text Matching Model [[paper]](https://arxiv.org/abs/2004.00390v1) [[code]](https://github.com/YuanEZhou/Grounded-Image-Captioning)  
  Yuanen Zhou, Meng Wang, Daqing Liu, Zhenzhen Hu, Hanwang Zhang
- Show, Edit and Tell: A Framework for Editing Image Captions [[paper]](https://arxiv.org/abs/2003.03107) [[code]](https://github.com/fawazsammani/show-edit-tell)  
  Fawaz Sammani, Luke Melas-Kyriazi
- Say As You Wish: Fine-Grained Control of Image Caption Generation With Abstract Scene Graphs [[paper]](https://arxiv.org/abs/2003.00387) [[code]](https://github.com/cshizhe/asg2cap)  
  Shizhe Chen, Qin Jin, Peng Wang, Qi Wu
- Normalized and Geometry-Aware Self-Attention Network for Image Captioning [[paper]](https://arxiv.org/abs/2003.08897)  
  Longteng Guo, Jing Liu, Xinxin Zhu, Peng Yao, Shichen Lu, Hanqing Lu
- Meshed-Memory Transformer for Image Captioning [[paper]](https://arxiv.org/abs/1912.08226) [[code]](https://github.com/aimagelab/meshed-memory-transformer)  
  Marcella Cornia, Matteo Stefanini, Lorenzo Baraldi, Rita Cucchiara
- X-Linear Attention Networks for Image Captioning [[paper]](https://arxiv.org/abs/2003.14080) [[code]](https://github.com/JDAI-CV/image-captioning)  
  Yingwei Pan, Ting Yao, Yehao Li, Tao Mei
- Transform and Tell: Entity-Aware News Image Captioning [[paper]](https://arxiv.org/abs/2004.08070) [[code]](https://github.com/alasdairtran/transform-and-tell) [[website]](https://transform-and-tell.ml/)  
  Alasdair Tran, Alexander Mathews, Lexing Xie

#### *`ACL 2020`*

* [MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning]() - Lei J et al, `ACL 2020`.
* [Dense-Caption Matching and Frame-Selection Gating for Temporal Localization in VideoQA](https://arxiv.org/pdf/2005.06409) - Kim H et al, `ACL 2020`.
- Clue: Cross-modal Coherence Modeling for Caption Generation [[paper]](https://arxiv.org/abs/2005.00908)  
  Malihe Alikhani, Piyush Sharma, Shengjie Li, Radu Soricut and Matthew Stone
- Improving Image Captioning Evaluation by Considering Inter References Variance[[paper]](https://www.aclweb.org/anthology/2020.acl-main.93/)  
  Yanzhi Yi, Hangyu Deng and Jinglu Hu  
- Improving Image Captioning with Better Use of Caption[[paper]](https://www.aclweb.org/anthology/2020.acl-main.664/)  
  Zhan Shi, Xu Zhou, Xipeng Qiu and Xiaodan Zhu

## Dataset
* [nocaps](https://nocaps.org/), LANG: `English`
* [MS COCO](http://cocodataset.org/), LANG: `English`.
* [Flickr 8k](https://forms.illinois.edu/sec/1713398), LANG: `English`.
* [Flickr 30k](http://shannon.cs.illinois.edu/DenotationGraph/), LANG: `English`.
* [AI Challenger](https://challenger.ai/dataset/caption), LANG: `Chinese`.
* [Visual Genome](http://visualgenome.org/), LANG: `English`.
* [SBUCaptionedPhotoDataset](http://www.cs.virginia.edu/~vicente/sbucaptions/), LANG: `English`.
* [IAPR TC-12](https://www.imageclef.org/photodata), LANG: `English, German and Spanish`.

## Image Captioning Challenge
* [Microsoft COCO Image Captioning](https://competitions.codalab.org/competitions/3221)
* [Google AI Blog: Conceptual Captions](http://ai.googleblog.com/2018/09/conceptual-captions-new-dataset-and.html)

## Popular Implementations
### PyTorch
* [ruotianluo/self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch)
* [ruotianluo/ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch)
* [jiasenlu/NeuralBabyTalk](https://github.com/jiasenlu/NeuralBabyTalk)
### TensorFlow
* [tensorflow/models/im2txt](https://github.com/tensorflow/models/tree/master/research/im2txt)
* [DeepRNN/image_captioning](https://github.com/DeepRNN/image_captioning)
### Torch
* [jcjohnson/densecap](https://github.com/jcjohnson/densecap)
* [karpathy/neuraltalk2](https://github.com/karpathy/neuraltalk2)
* [jiasenlu/AdaptiveAttention](https://github.com/jiasenlu/AdaptiveAttention)
### Others
* [emansim/text2image](https://github.com/emansim/text2image)
* [apple2373/chainer-caption](https://github.com/apple2373/chainer-caption)
* [peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)

## Licenses

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Zhihong Chen](https://github.com/zhjohnchan) has waived all copyright and related or neighboring rights to this work.

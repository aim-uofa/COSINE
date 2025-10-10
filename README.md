<div align="center">

<h1>Unified Open-World Segmentation with Multi-Modal Prompts </h1>

[Yang Liu](https://scholar.google.com/citations?user=9JcQ2hwAAAAJ&hl=en)<sup>1*</sup>, &nbsp; 
Yufei Yin<sup>2*</sup>, &nbsp; 
Chenchen Jing<sup>3</sup>, &nbsp; 
Muzhi Zhu<sup>1</sup>, &nbsp;
[Hao Chen](https://stan-haochen.github.io/)<sup>1</sup>, &nbsp;
Yuling Xi<sup>1</sup>, &nbsp;
Bo Feng<sup>4</sup>, &nbsp;
Hao Wang<sup>4</sup>, &nbsp;
Shiyu Li<sup>4</sup>, &nbsp;
[Chunhua Shen](https://cshen.github.io/)<sup>1</sup>

<sup>1</sup>[Zhejiang University](https://www.zju.edu.cn/english/), &nbsp;
<sup>2</sup>Hangzhou Dianzi University, &nbsp;
<sup>3</sup>Zhejiang University of Technology, &nbsp;
<sup>4</sup>Apple


</div>

## 🚀 Overview
<div align="center">
<img width="800" alt="image" src="figs/teaser.png">
</div>


## 📖 Description

In this work, we present COSINE, a unified open-world segmentation model that consolidates open-vocabulary segmentation and in-context segmentation with multi-modal prompts (e.g. text and image). 
COSINE exploits foundation models to extract representations for an input image and corresponding multi-modal prompts, and a SegDecoder to align these representations, model their interaction, and obtain masks specified by input prompts across different granularities.
In this way, COSINE overcomes architectural discrepancies, divergent learning objectives, and distinct representation learning strategies of previous pipelines for open-vocabulary segmentation and in-context segmentation.
Comprehensive experiments demonstrate that COSINE has significant performance improvements in both open-vocabulary and in-context segmentation tasks. 
Our exploratory analyses highlight that the synergistic collaboration between using visual and textual prompts leads to significantly improved generalization over single-modality approaches. 

## 🎫 License

For academic use, this project is licensed under [the 2-clause BSD License](https://opensource.org/license/bsd-2-clause). 
For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).
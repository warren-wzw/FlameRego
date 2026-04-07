# Introduce
This repository contains the source code for our paper:

[面向复杂燃烧状态识别的多编码器特征融合模型]([https://www.mdpi.com/1996-1073/17/20/5143](https://kns.cnki.net/kcms2/article/abstract?v=Jz5IuRg0t03-cDqcifcRq6-nls248fUhuHJ5p_Zi3QDebkxYyU2LI3hCN6pl_RRUlbMtUQWF7UYwf0yGYJGd794uXIQmmpYE577SeNu1qXKomemEu2KK28oANk5mzhlefaD_TVs9TPtNA1TNGDAnI4nbbvmuGsNcPCD6dLPTHrxScNaUtK40mw==&uniplatform=NZKPT))

## Functions
* It introduces a self-attention mechanism in the skip connections to enhance the extraction of key information from deep features;
* It replaces traditional convolutions with depthwise separable convolutions to reduce the model’s computational complexity and improve inference speed;
* It substitutes the bottleneck layer with a Transformer encoder to leverage the global modeling capabilities of Transformers, allowing the model to better understand contextual information within the image.

## Model Architecture
![](./images/1.png)

## Results
![](./images/2.png)
## Env set
* cuda               11.8
* python             3.8.0
* conda create --name StrawSeg python=3.8.0
* Detail please reference requierment.txt

## About author
* warren@伟
* Blog：[CSDN](https://blog.csdn.net/warren103098?type=blog)

## If this work is helpful to you, please cite it as：
```
[1]王志伟,何德峰,李廉明,等.面向复杂燃烧状态识别的多编码器特征融合模型[C]//中国自动化学会.2024中国自动化大会论文集.浙江工业大学信息工程学院;全省清洁能源转化与利用重点实验室;嘉兴新嘉爱斯热电有限公司;浙江大学能源高效清洁利用全国重点实验室;,2024:383-388.DOI:10.26914/c.cnkihy.2024.052399.
```

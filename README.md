
## [MedicalSAC: Segment Any State Change in Medical Images]
Xiaowei Zhao, Guyue Hu, Anwei Jiang, Chenglong Li, Jin Tang

## Introduction
For long-term therapeutic conditions such as brain tumors and breast cancer, different subregions within a lesion often exhibit heterogeneous tolerance and response to chemotherapy or radiotherapy, leading to distinct temporal lesion state changes, such as localized deterioration or recovery. Such treatment-response heterogeneity is a major contributor to treatment failure and recurrence. Therefore, making timely identification of these state changes is clinically essential for precise adjustment of therapeutic strategies. In this paper, we introduce a new task called medical state change detection that jointly depicts lesion morphology, location, and state changes. However, temporal misalignment between the image and the need to capture both subtle changes and stable anatomical cues makes lesion state detection highly challenging. Meanwhile, the lack of annotated medical change detection datasets further limits research advances. To address these challenges, we propose a novel Segment Any State Change Model, termed MedicalSAC, to simultaneously utilize the temporal context and feature differences for precise medical state change detection. In particular, we introduce a temporal context enhancement adapter to model cross-temporal relationships, and a bidirectional difference refinement module to effectively capture temporal variations. To provide a comprehensive evaluation platform for medical state change detection, we also create a benchmark dataset including 2,839 temporal image pairs with two medical imaging modalities, four lesion types, and three pixel-wise state change labels. Extensive experiments demonstrate the effectiveness and superiority of our method in medical state change detection tasks.

## Framework
<div align="center">
<iframe 
src="[https://github.com/XWei98/MedicalSAC/edit/main/intro.pdf](https://github.com/XWei98/MedicalSAC/blob/main/intro.pdf)"
width="1000" 
height="700"
frameborder="0">
</iframe>
</div>


## Requirements
Our project does not depend on installing SAM2. If you have already configured an environment for SAM2, then directly using this environment should also be fine. You may also create a new conda environment:

```shell
conda create -n sam2-unet python=3.10
conda activate sam2-unet
pip install -r requirements.txt
```


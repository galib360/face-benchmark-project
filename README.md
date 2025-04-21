# **"Wild West" of Evaluating Speech-Driven 3D Facial Animation Synthesis: A Benchmark Study <br> (Accepted at Eurographics 2025, London, UK)**
### This repository contains the documentation and edited code scripts of models used for the benchmark study in the paper. 

> **"Wild West" of Evaluating Speech-Driven 3D Facial Animation Synthesis: A Benchmark Study (Accepted at [Eurographics 2025](https://eg25.cs.ucl.ac.uk/main/home.html), London, UK)**
>
> <a href='http://dx.doi.org/10.1111/cgf.70073'><img src='https://img.shields.io/badge/Paper-orange'></a>
> <a href='https://galib360.github.io/face-benchmark-project/'><img src='https://img.shields.io/badge/Project-Website-blue'></a>
> <a href='https://galib360.github.io/face-benchmark-project/#video-container' target="_blank"><img src='https://img.shields.io/badge/Supplementary-Video-Green'></a> 
> 
> ### Abstract <br>
>Recent advancements in the field of audio-driven 3D facial animation have accelerated rapidly, with numerous papers being published in a short span of time. This surge in research has garnered significant attention from both academia and industry with its potential applications on digital humans. Various approaches, both deterministic and non-deterministic, have been explored based on foundational advancements in deep learning algorithms. However, there remains no consensus among researchers on standardized methods for evaluating these techniques. Additionally, rather than converging on a common set of datasets and objective metrics suited for specific methods, recent works exhibit considerable variation in experimental setups. This inconsistency complicates the research landscape, making it difficult to establish a streamlined evaluation process and rendering many cross-paper comparisons challenging. Moreover, the common practice of A/B testing in perceptual studies focus only on two common metrics and not sufficient for non-deterministic and emotion-enabled approaches. The lack of correlations between subjective and objective metrics points out that there is a need for critical analysis in this space.  In this study, we address these issues by benchmarking state-of-the-art deterministic and non-deterministic models, utilizing a consistent experimental setup across a carefully curated set of objective metrics and datasets. We also conduct a perceptual user study to assess whether subjective perceptual metrics align with the objective rankings. Our findings indicate that model rankings do not necessarily generalize across datasets, and subjective metric ratings are not always consistent with their corresponding objective metrics. The code documentation together with the edited scripts related to this benchmark study are made publicly available here.


<p align="center">
  <a href='https://galib360.github.io/face-benchmark/#video-container' target="_blank"><img src="./assets/SupplementaryVideo.gif"/></a>
</p>

## **Datasets**
<details><summary>Click to expand</summary>

### BIWI
Request and download [BIWI](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html) dataset following the link. The data preparation for BIWI is borrowed from [CodeTalker](https://github.com/Doubiiu/CodeTalker/tree/main/BIWI).


### Multiface
[Multiface](https://github.com/facebookresearch/multiface) dataset can be accessed following the link. The data preparation for Multiface is borrowed from [FaceDiffuser](https://github.com/uuembodiedsocialai/FaceDiffuser/tree/main/data/multiface).

### 3DMEAD
Download 3DMEAD dataset following the instruction of [EMOTE](https://github.com/radekd91/inferno/tree/release/EMOTE/inferno_apps/TalkingHead/data_processing). This dataset represents facial animations using FLAME parameters.

#### Data Download and Preprocess 
- We use the data preprocessing workflow as done in [ProbTalk3D](https://github.com/uuembodiedsocialai/ProbTalk3D)
- Please refer to the `README.md` file in `datasets/3DMEAD_preprocess/` folder in the [ProbTalk3D](https://github.com/uuembodiedsocialai/ProbTalk3D/tree/main/datasets/3DMEAD_preprocess) repository. 
- The dataset split also follows the split proposed in [ProbTalk3D](https://github.com/uuembodiedsocialai/ProbTalk3D/tree/main/datasets/mead-splits-pred).
- For the sentence level splits, we refer to [this code script](https://github.com/uuembodiedsocialai/ProbTalk3D/blob/c7d157a6a17676f15414912acc93075bf7e9e981/framework/data/mead.py#L166)

</details>

## **FaceFormer**
<details><summary>Click to expand</summary>

### Model
- FaceFormer model is available publicly through [this link](https://github.com/EvelynFan/FaceFormer/tree/main).
- Our study trained FaceFormer on BIWI, Multiface and 3DMEAD datasets. 
- The edited code scripts for respective datasets can be found in `./Datasets/` directory. Change your paths accordingly in the script(s). 
- Replace the code scripts (or edit your script according to the provided script) train on specific dataset.


### Hyperparameters
The hyperparameters of the model is kept the same as the original proposed model. 

</details>

## **CodeTalker**
<details><summary>Click to expand</summary>

### Model
- CodeTalker model is available publicly through [this link](https://github.com/Doubiiu/CodeTalker/tree/main).
- Our study trained CodeTalker on BIWI, Multiface and 3DMEAD datasets. 
- The edited code scripts for respective datasets can be found in `./Datasets/` directory. Change your paths accordingly in the script(s). 
- Replace the code scripts (or edit your script according to the provided script) train on specific dataset.
- For original CodeTalker (deterministic), do not use the provided edited `quantizer.py` script. 
- For CodeTalker-ND, use the provided edited `quantizer.py` script.


### Hyperparameters
The hyperparameters of the model is kept the same as the original proposed model. 

</details>


## **FaceXHuBERT**
<details><summary>Click to expand</summary>

### Model
- FaceXHuBERT model is available publicly through [this link](https://github.com/galib360/FaceXHuBERT).
- Our study trained FaceXHuBERT on BIWI, Multiface and 3DMEAD datasets. 
- The edited code scripts for respective datasets can be found in `./Datasets/` directory. Change your paths accordingly in the script(s). 
- Replace the code scripts (or edit your script according to the provided script) train on specific dataset.

### Hyperparameters
- The hyperparameters of the model is kept the same as the original proposed model. 

</details>


## **FaceDiffuser**
<details><summary>Click to expand</summary>

### Model
- FaceDiffuser model is available publicly through [this link](https://github.com/uuembodiedsocialai/FaceDiffuser).
- Our study trained FaceDiffuser on BIWI, Multiface and 3DMEAD datasets. 
- The edited code scripts for respective datasets can be found in `./Datasets/` directory. Change your paths accordingly in the script(s). 
- Replace the code scripts (or edit your script according to the provided script) train on specific dataset.


### Hyperparameters
- The hyperparameters of the model is kept the same as the original proposed model. 
- For BIWI and Multiface, V-FaceDiffuser config is used.
- For 3DMEAD dataset, B-FaceDiffuser config is used.

</details>


## **ProbTalk3D**
<details><summary>Click to expand</summary>

### Model
- ProbTalk3D model is available publicly through [this link](https://github.com/uuembodiedsocialai/ProbTalk3D).
- Our study trained ProbTalk3D on 3DMEAD dataset.
- For training, we refer to the original model repository. 


### Hyperparameters
The hyperparameters of the model is kept the same as the original proposed model. 

</details>


## **Training details**
<details><summary>Click to expand</summary>

### Hardware
- The models used in this study were trained on a shared compute cluster running - 
- Linux with AMD EPYC 7313 CPU 
- Nvidia A16 GPU, 1TB RAM.

### Hyperparameters
- For the hyperparameters for specific models, we refer to the code scripts in `Datasets-Models/` directory

### Datasets and pre-processing
- For dataset preprocessing, we refer to the documentation in READMEs in `Datasets-Models/<dataset>/` directories

### Hyperparameters
The hyperparameters of the model is kept the same as the original proposed model. 

</details>



## **Evaluation**
<details><summary>Click to expand</summary>

### Quantitative Evaluation
The objective metrics used in our benchmark study are carefully selected to cover the most used metrics that have publicly available code implementations. We refer to `./Evaluaton/README.md` for further information. 

</details>




## Citation ## 
If you find this repository useful for your research work, please consider starring this repository and citing it:
```
@article{Haque2025,
  title = {“Wild West” of Evaluating Speech‐Driven 3D Facial Animation Synthesis: A Benchmark Study},
  ISSN = {1467-8659},
  url = {http://dx.doi.org/10.1111/cgf.70073},
  DOI = {10.1111/cgf.70073},
  journal = {Computer Graphics Forum},
  publisher = {Wiley},
  author = {Haque,  Kazi Injamamul and Pavlou,  Alkiviadis and Yumak,  Zerrin},
  year = {2025},
  month = apr 
} 

```

## **Acknowledgements**

We thank the authors of [FaceFormer](https://github.com/EvelynFan/FaceFormer), [CodeTalker](https://github.com/Doubiiu/CodeTalker), [FaceXHuBERT](https://github.com/galib360/FaceXHuBERT), [FaceDiffuser](https://github.com/uuembodiedsocialai/FaceDiffuser), [ProbTalk3D](https://github.com/uuembodiedsocialai/ProbTalk3D) and [EMOTE](https://emote.is.tue.mpg.de/). We appreciate the authors for making their code and/or dataset(s) available that facilitates open research.

Any third-party packages are owned by their respective authors and must be used under their respective licenses.

## **License**
This repository is released under [CC-BY-NC-4.0-International License](https://github.com/Gibberlings3/GitHub-Templates/blob/master/License-Templates/CC-BY-NC-4.0/LICENSE-CC-BY-NC-4.0.md).

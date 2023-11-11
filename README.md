# AViTMP: Exploiting Image-Related Inductive Biases in Single-Branch Visual Tracking. 

The official implement of AViTMP [_Arxiv_](https://arxiv.org/abs/2310.19542)  


<p align="center">
  <img width="85%" src="assets/framework.png" alt="Framework"/>
</p>

<p align="center">
    <img width="50%" src="assets/cycletrack.png" alt="cycletrack"/>
</p>

## TL;DR
 To tackle the inferior effectiveness of the vanilla ViT, we propose an Adaptive ViT Model Prediction tracker (AViTMP) to bridge the gap between single-branch network and discriminative models. Specifically, in the proposed encoder AViT-Enc, we introduce an adaptor module and joint target state embedding to enrich the dense embedding paradigm based on ViT. Then, we combine AViT-Enc with a dense-fusion decoder and a discriminative target model to predict accurate location. Further, to mitigate the limitations of conventional inference practice, we present a novel inference pipeline called CycleTrack, which bolsters the tracking robustness in the presence of distractors via bidirectional cycle tracking verification. Lastly, we propose a dual-frame update inference strategy that adeptively handles significant challenges in long-term scenarios. 
 


| Tracker |      LaSOT_Ext   | LaSOT   | AVisT            | VOT2020     | UAV123 |    TNL2k |  TrackingNet |  OTB100 | NFS30   |
|:------------:|:-----------:|:-----------:|:-----------------:|:-----------:|:--------------:|:----------:|:---------:|:---------:|:---------:|
|  AViTMP (AUC)    |   50.2      |    70.7     |       54.9        |    31.4    |      70.1      |    54.5    |   82.8     |   70.3   |    66.3 |



## Setup Environment
 
```
conda create --name pytracking --file requirements.txt
source activate pytracking
```

```
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
cd ltr
```

## Data Prapare and model training 
1. link datasets into './data'
```
     |--data
        |--avist
        |--coco
        |--got10k
        |--lasot
        |--uav
        |--lasot
        |--trackingnet
        |--.......
```
2. download pretrained ViT-B model into ['./pretrained_model'](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)
2. Training  AViTMP model. 

```
cd ltr
CUDA_VISIBLE_DEVICES=5,6,7,4  python run_training.py
# if need save the training logs into txt file. 
 nohup python -u run_training.py  >trainlog.txt 2>&1 &  
```



## Infenrence
### 1. put the [checkpoint](too big to upload) into './pytracking/networks/'
### 2. Evalute for datasets: lasot, lasot_extension_subset, avist, uav, trackingnet, nfs, otb, tnl2k.  
All evaluated results are put in './tracking_results'
```
python run_tracker.py --dataset_name uav   
```
### 3. Evalute for VOT2020 (following [VOT-toolkit guidence](https://votchallenge.net/howto/))
```
#1. env setting for VOT2020， 
vot-toolkit==0.5.3 
vot-trax==3.0.2

3. commends of evaluate. 
vot initialize  AViTMP  -workspace vot2020
vot evaluate --workspace vot2020 AViTMP
vot analysis --workspace vot2020 AViTMP 
```
### 4. Evalute for VOTS2023 (multi-object tracking \& segmentation)
1. install segmentation model [SAM-HQ](https://github.com/SysCV/sam-hq.git)

2. combine VOT with segmentation method to build a tracking \& segmentation two-stage method.
```
# Note: env setting for VOTS2023
vot-toolkit==0.6.4
vot-trax==4.0.1
```

3. inference to generate target mask.


## Acknowledgement

This is a combination version of the python tracking framework [PyTracking](https://github.com/visionml/pytracking) 
and [OSTrack](https://github.com/botaoye/OSTrack.git).  





### Cite
```
@misc{tang2023exploiting,
      title={Exploiting Image-Related Inductive Biases in Single-Branch Visual Tracking}, 
      author={Chuanming Tang and Kai Wang and Joost van de Weijer and Jianlin Zhang and Yongmei Huang},
      year={2023},
      eprint={2310.19542},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
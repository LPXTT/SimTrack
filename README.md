SimTrack
=========================================
Official Code for [Backbone is All Your Need: A Simplified Architecture for Visual Object Tracking](https://arxiv.org/abs/2107.02960) accepted by ECCV 2022.
This is the first version of our code.
We initially implement SimTrack in [Pysot](https://github.com/STVIR/pysot), which is adopted by the Sensetime company. 
There are several private distribution and data loading codes which can not be published.
Therefore, I am changing the pipeline to the public reposity [STARK](https://github.com/researchmm/Stark).
It will take some time to fully convert the code from [Pysot](https://github.com/STVIR/pysot) to [STARK](https://github.com/researchmm/Stark).
For now, we first release the testing code. The training code will come as soon as possible. Thanks for your understanding!

## Requirements
- torch==1.8.1
- torchvision==0.9.0
- timm==0.5.4


## Results (AUC)
|method|  FLOPs    |   LaSOT | TNL2K | TrackingNet | GOT-10k_Test | UAV123  | model|
|:------:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:------:|
|SimTrack| 25.0G | 69.3 | 54.8 | 82.3 | 70.6 | 69.8| [Sim-B/16](https://drive.google.com/file/d/19iSJi14yfJsi_XN5bfKdkBPUHgFzagg9/view?usp=sharing)|


## Evaluation
Download the model [Sim-B/16](https://drive.google.com/file/d/19iSJi14yfJsi_XN5bfKdkBPUHgFzagg9/view?usp=sharing). Add the model path to https://github.com/LPXTT/SimTrack/blob/a238932fd0cba9aa4a6fcdb590470d5882e5b0b4/lib/test/tracker/simtrack.py#L19
```
python tracking/test.py simtrack baseline --dataset got10k_test --threads 32
```

## Training
If you are in a hurry, you can try to train a model before our final version. I think there is no tricky bug.
```
python tracking/train.py --script simtrack --config baseline_got10k_only --save_dir . --mode multiple --nproc_per_node 8
```

## Thanks
This implementation is based on [STARK](https://github.com/researchmm/Stark). Please ref to their reposity for more details.

## Citation
If you find that this project helps your research, please consider citing our paper:
```
@article{chen2022backbone,
  title={Backbone is All Your Need: A Simplified Architecture for Visual Object Tracking},
  author={Chen, Boyu and Li, Peixia and Bai, Lei and Qiao, Lei and Shen, Qiuhong and Li, Bo and Gan, Weihao and Wu, Wei and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2203.05328},
  year={2022}
}

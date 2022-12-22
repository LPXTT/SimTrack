SimTrack
=========================================
Official Code for [Backbone is All Your Need: A Simplified Architecture for Visual Object Tracking](https://arxiv.org/abs/2107.02960) accepted by ECCV 2022.


## Requirements
- python==3.8.15
- torch==1.8.1
- torchvision==0.9.0
- timm==0.5.4


## Results (AUC)
|method|  FLOPs    |   LaSOT | TNL2K | TrackingNet | GOT-10k_Test | UAV123  | model|
|:------:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:------:|
|SimTrack| 25.0G | 69.3 | 54.8 | 82.3 | 70.6 | 69.8| [Sim-B/16](https://drive.google.com/file/d/19iSJi14yfJsi_XN5bfKdkBPUHgFzagg9/view?usp=sharing)|
|Raw Results| - | [LaSOT](https://drive.google.com/file/d/1bVohxZGlpdTmEwIm0IRB9vbM6hIZOKpy/view?usp=sharing) | [TNL2K](https://drive.google.com/file/d/1B9Y3QDBWL16ku5BpavharMdfqVQvofhF/view?usp=sharing) | [TrackingNet](https://drive.google.com/file/d/1nnQqXN4BkUd6CORieHmGuTKSvo0rAZAZ/view?usp=sharing) | [GOT-10k_Test](https://drive.google.com/file/d/1G5HgEUUkx8EWglvTFpZrJ5plKDqHCF9X/view?usp=sharing) | [UAV123](https://drive.google.com/file/d/1U6SnBZLMqgPqFv-Gg0TvP6dtserjo5RA/view?usp=sharing) | - |


## Evaluation
Download the model [Sim-B/16](https://drive.google.com/file/d/19iSJi14yfJsi_XN5bfKdkBPUHgFzagg9/view?usp=sharing). Add the model path to https://github.com/LPXTT/SimTrack/blob/a238932fd0cba9aa4a6fcdb590470d5882e5b0b4/lib/test/tracker/simtrack.py#L19
```
python tracking/test.py simtrack baseline --dataset got10k_test --threads 32
```

## Training
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

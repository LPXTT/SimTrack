SimTrack
=========================================
Official Code for [Backbone is All Your Need: A Simplified Architecture for Visual Object Tracking](https://arxiv.org/abs/2107.02960) accepted by ECCV 2022.
This is the first version of our code.
We initially implement SimTrack in [Pysot] (https://github.com/STVIR/pysot), which is adopted by the Sensetime company. 
There are several private distribution and data loading codes which can not be published.
Therefore, I am changing the pipeline to the public reposity [STARK](https://github.com/researchmm/Stark).
It will take some time to fully convert the code from [Pysot](https://github.com/STVIR/pysot) to [STARK](https://github.com/researchmm/Stark).
For now, we first release the testing code. The training code will come as soon as possible. Thank you for your understanding!

## Requirements
- torch==1.8.1
- torchvision==0.9.0
- timm==0.5.4

The requirements.txt file lists other Python libraries that this project depends on, and they will be installed using:
pip3 install -r requirements.txt

## Results (AUC)
|method|  FLOPs    |   LaSOT | TNL2K | TrackingNet | GOT-10k_Test | UAV123  | model|
|:------:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:------:|
|SimTrack| 25.0G | 69.3 | 54.8 | 82.3 | 70.6 | 69.8| [Sim-B/16](https://drive.google.com/file/d/1ryxn9TEwnoDTTQxv5JMyWpvU2OuOMLqL/view?usp=sharing)|


## Training
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model glit_tiny_patch16_224 --clip-grad 1.0 --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save
```

## Evaluation
Download the model [Glit_Tiny](https://drive.google.com/file/d/1ryxn9TEwnoDTTQxv5JMyWpvU2OuOMLqL/view?usp=sharing)
```
python main.py --eval --resume glit_tiny.pth.tar --data-path /path/to/imagenet

```

## Thanks
This implementation is based on [STARK](https://github.com/researchmm/Stark). Please ref to their reposity for more details.

## Citation

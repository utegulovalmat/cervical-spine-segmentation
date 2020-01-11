### Cervical spine segmentation

Results for models:


|Model            |Encoder           |N4 correction  |DSC, %   |
|-----------------|:----------------:|:-------------:|:-------:|
|Unet             |random init.      |Yes            |--       |
|Unet             |random init.      |No             |84       |
|Unet             |resnet50          |Yes            |--       |
|Unet             |resnet50          |No             |84       |
|Unet             |inceptionv4       |Yes            |--       |
|Unet             |inceptionv4       |No             |80       |
|FPN              |random init.      |Yes            |--       |
|FPN              |random init.      |No             |83       |
|FPN              |resnet50          |Yes            |--       |
|FPN              |resnet50          |No             |80       |
|FPN              |inceptionv4       |Yes            |--       |
|FPN              |inceptionv4       |No             |79       |
|Linknet          |random init.      |Yes            |--       |
|Linknet          |random init.      |No             |84       |
|Linknet          |resnet50          |Yes            |--       |
|Linknet          |resnet50          |No             |86       |
|Linknet          |inceptionv4       |Yes            |--       |
|Linknet          |inceptionv4       |No             |82       |
|PSPNet           |random init.      |Yes            |--       |
|PSPNet           |random init.      |No             |82       |
|PSPNet           |resnet50          |Yes            |--       |
|PSPNet           |resnet50          |No             |74       |
|PSPNet           |inceptionv4       |Yes            |--       |
|PSPNet           |inceptionv4       |No             |74       |




To reproduce results run one of:

```bash
# train using axis 0 - all volumes
nohup python -m segmentation_models_pytorch.experiments.train_model -in path/to/nrrd/dataset --train_all all --extract_slices 1 --use_axis 0 &

# train using axis 1 - all volumes
nohup python -m segmentation_models_pytorch.experiments.train_model -in path/to/nrrd/dataset --train_all all --extract_slices 1 --use_axis 1 &

# train using axis 2 - all volumes
nohup python -m segmentation_models_pytorch.experiments.train_model -in path/to/nrrd/dataset --train_all all --extract_slices 1 --use_axis 2 &

# train using axis 012 - all volumes
nohup python -m segmentation_models_pytorch.experiments.train_model -in path/to/nrrd/dataset --train_all all --extract_slices 1 --use_axis 012 &
```

Pipeline of models in the training queue is configured in `segmentation_models_pytorch/experiments/pipeline.csv`

### Models <a name="models"></a>

#### Architectures <a name="architectires"></a>
 - [Unet](https://arxiv.org/abs/1505.04597)
 - [Linknet](https://arxiv.org/abs/1707.03718)
 - [FPN](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
 - [PSPNet](https://arxiv.org/abs/1612.01105)

#### Encoders <a name="encoders"></a>
`cervical-spine-segmentation/segmentation_models_pytorch/encoders`

### References
Code extended from: https://github.com/qubvel/segmentation_models.pytorch

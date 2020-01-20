### Cervical spine segmentation

Trained model can be downloaded here: 
https://mtixnat.uni-koblenz.de/owncloud/index.php/s/UBkFmvpQpzaXqj5

Results for models trained on SAGITTAL view (axis-0):

|Model            |Encoder           |N4 correction  |DSC, %   |
|-----------------|:----------------:|:-------------:|:-------:|
|Unet             |random init.      |Yes            |75       |
|Unet             |random init.      |No             |84       |
|Unet             |resnet50          |Yes            |87       |
|Unet             |resnet50          |No             |84       |
|Unet             |inceptionv4       |Yes            |79       |
|Unet             |inceptionv4       |No             |80       |
|FPN              |random init.      |Yes            |83       |
|FPN              |random init.      |No             |83       |
|FPN              |resnet50          |Yes            |86       |
|FPN              |resnet50          |No             |80       |
|FPN              |inceptionv4       |Yes            |79       |
|FPN              |inceptionv4       |No             |79       |
|Linknet          |random init.      |Yes            |81       |
|Linknet          |random init.      |No             |84       |
|Linknet          |resnet50          |Yes            |86       |
|Linknet          |resnet50          |No             |86       |
|Linknet          |inceptionv4       |Yes            |--       |
|Linknet          |inceptionv4       |No             |82       |
|PSPNet           |random init.      |Yes            |79       |
|PSPNet           |random init.      |No             |82       |
|PSPNet           |resnet50          |Yes            |80       |
|PSPNet           |resnet50          |No             |74       |
|PSPNet           |inceptionv4       |Yes            |78       |
|PSPNet           |inceptionv4       |No             |74       |


Results for models trained on CORONAL view (axis-1):

|Model            |Encoder           |N4 correction  |DSC, %   |
|-----------------|:----------------:|:-------------:|:-------:|
|Unet             |random init.      |Yes            |       |
|Unet             |random init.      |No             |       |
|Unet             |resnet50          |Yes            |       |
|Unet             |resnet50          |No             |       |
|Unet             |inceptionv4       |Yes            |       |
|Unet             |inceptionv4       |No             |       |
|FPN              |random init.      |Yes            |       |
|FPN              |random init.      |No             |       |
|FPN              |resnet50          |Yes            |       |
|FPN              |resnet50          |No             |       |
|FPN              |inceptionv4       |Yes            |       |
|FPN              |inceptionv4       |No             |       |
|Linknet          |random init.      |Yes            |       |
|Linknet          |random init.      |No             |       |
|Linknet          |resnet50          |Yes            |       |
|Linknet          |resnet50          |No             |       |
|Linknet          |inceptionv4       |Yes            |       |
|Linknet          |inceptionv4       |No             |       |
|PSPNet           |random init.      |Yes            |       |
|PSPNet           |random init.      |No             |       |
|PSPNet           |resnet50          |Yes            |       |
|PSPNet           |resnet50          |No             |       |
|PSPNet           |inceptionv4       |Yes            |       |
|PSPNet           |inceptionv4       |No             |       |

Results for models trained on AXIAL view (axis-2):

|Model            |Encoder           |N4 correction  |DSC, %   |
|-----------------|:----------------:|:-------------:|:-------:|
|Unet             |random init.      |Yes            |       |
|Unet             |random init.      |No             |       |
|Unet             |resnet50          |Yes            |       |
|Unet             |resnet50          |No             |       |
|Unet             |inceptionv4       |Yes            |       |
|Unet             |inceptionv4       |No             |       |
|FPN              |random init.      |Yes            |       |
|FPN              |random init.      |No             |       |
|FPN              |resnet50          |Yes            |       |
|FPN              |resnet50          |No             |       |
|FPN              |inceptionv4       |Yes            |       |
|FPN              |inceptionv4       |No             |       |
|Linknet          |random init.      |Yes            |       |
|Linknet          |random init.      |No             |       |
|Linknet          |resnet50          |Yes            |       |
|Linknet          |resnet50          |No             |       |
|Linknet          |inceptionv4       |Yes            |       |
|Linknet          |inceptionv4       |No             |       |
|PSPNet           |random init.      |Yes            |       |
|PSPNet           |random init.      |No             |       |
|PSPNet           |resnet50          |Yes            |       |
|PSPNet           |resnet50          |No             |       |
|PSPNet           |inceptionv4       |Yes            |       |
|PSPNet           |inceptionv4       |No             |       |

To reproduce results run one of:

```bash
# train using axis 0
nohup python -m segmentation_models_pytorch.experiments.train_model -in path/to/nrrd/dataset --train_all all --extract_slices 1 --use_axis 0 &

# train using axis 1
nohup python -m segmentation_models_pytorch.experiments.train_model -in path/to/nrrd/dataset --train_all all --extract_slices 1 --use_axis 1 &

# train using axis 2
nohup python -m segmentation_models_pytorch.experiments.train_model -in path/to/nrrd/dataset --train_all all --extract_slices 1 --use_axis 2 &

# train using axis 012 
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

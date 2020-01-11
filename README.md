Code extended from: https://github.com/qubvel/segmentation_models.pytorch

### Cervical spine segmentation

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
List of encoders: https://github.com/qubvel/segmentation_models.pytorch

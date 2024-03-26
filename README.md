# Object-centric extensions to Projected GAN

This repo extends the codebase [ProjectedGAN](https://github.com/autonomousvision/projected-gan) by [Axel Sauer](https://axelsauer.com/), [Kashyap Chitta](https://kashyap7x.github.io/), [Jens Müller](https://hci.iwr.uni-heidelberg.de/users/jmueller), and [Andreas Geiger](http://www.cvlibs.net/).

- `pg_modules/slate_transformer.py` and `pg_modules/slate_utils.py` have been taken from `https://github.com/singhgautam/slate`.
- `pg_modules/slot_attention.py` has been taken from `https://github.com/evelinehong/slot-attention-pytorch`. The `eval_slots_*.ipynb` notebooks have also been build on top of that repo.
- `pg_modules/dinosaur.py` has been built based on the paper from `https://arxiv.org/abs/2209.14860`.

Note that the StyleGAN generator is not supported, but still included, so that it could be extended.

This is the direct conditioning version.


## Requirements ##
- The code was tested with `python 3.9.7`
- Use the following commands with Miniconda3 to create and activate your PG Python environment:
  - ```conda env create -f environment.yml```
  - ```conda activate pg```
  - ```pip install git+https://github.com/openai/CLIP.git```
- You will need a pretrained DINOSAUR model. Such a pretrained model on the [COCO](https://cocodataset.org/#home) dataset are already included in this repo.


## Data Preparation ##

This section is unchanged from `ProjectedGAN`

For a quick start, you can download the few-shot datasets provided by the authors of [FastGAN](https://github.com/odegeasslbc/FastGAN-pytorch). You can download them [here](https://drive.google.com/file/d/1aAJCZbXNHyraJ6Mi13dSbe7pTyfPXha0/view). To prepare the dataset at the respective resolution, run for example
```
python dataset_tool.py --source=./data/pokemon --dest=./data/pokemon256.zip \
  --resolution=256x256 --transform=center-crop
```
You can get the datasets we used in our paper at their respective websites: 

[CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/), [FFHQ](https://github.com/NVlabs/ffhq-dataset), [Cityscapes](https://www.cityscapes-dataset.com/), [LSUN](https://github.com/fyu/lsun), [AFHQ](https://github.com/clovaai/stargan-v2), [Landscape](https://www.kaggle.com/arnaud58/landscape-pictures).

## Training ##

For the training commands have a look at the `sbatches` folder. The contained `.sbatch` files contain training commands for different datasets. The commands use a [singularity](https://docs.sylabs.io/guides/latest/user-guide/) image. If you do not want to use singularity, just run the pure python command.

## Evaluation

Take a look at the notebooks `eval_slots.ipynb` and `eval_slots_clevr.ipnb`. They require folder based datasets, rather than the `.zip` file created in the data preparation step.

Just set the `BASE_PATH` to the path where the training output is (by default in `./training-runs`), then also change the `dataset_path_by_name` dictionary to fit your dataset locations.

The interactive cell allows you to do compositions by replacing one slot by another. Sould you want to exchange more than one slot, write your interpolation into the `index == 7` case of `show_interpolation` and select `7` in the interactive cell.
 

## Acknowledgments ##

From the original ProjectedGAN:

> Our codebase build and extends the awesome [StyleGAN2-ADA repo](https://github.com/NVlabs/stylegan2-ada-pytorch) and [StyleGAN3 repo](https://github.com/NVlabs/stylegan3), both by Karras et al.
> Furthermore, we use parts of the code of [FastGAN](https://github.com/odegeasslbc/FastGAN-pytorch) and [MiDas](https://github.com/isl-org/MiDaS).

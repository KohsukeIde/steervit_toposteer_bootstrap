# Dataset Installation and Preparation

Please create an account on [ImageNet](https://www.image-net.org/) to download the dataset. Once you have the dataset, please follow the instructions below to prepare it.

### ImageNet-1K

The root directory of the dataset should hold the following contents. Download [`labels.txt`](assets/labels.txt) from the `assets/` folder and add it to your dataset directory as shown below:

<pre>
Imagenet-1K
├── train/
│ ├── n01440764/
│ │ └── n01440764_10026.JPEG
│ ├── [...]
│ └── n15075141/
│ └── n15075141_9993.JPEG
|
├── val/
│ ├── n01440764/
│ │ └── ILSVRC2012_val_00000293.JPEG
│ ├── [...]
│ └── n15075141/
│ └── ILSVRC2012_val_00049174.JPEG
|
└── labels.txt ← from data_prep/labels.txt
</pre>


The provided dataset implementation expects a few additional metadata files to be present under the extra directory:

<pre>
Imagenet-1K/extra/
├── class-ids-TRAIN.npy
├── class-ids-VAL.npy
├── class-names-TRAIN.npy
├── class-names-VAL.npy
├── entries-TEST.npy
├── entries-TRAIN.npy
└── entries-VAL.npy
</pre>
These metadata files can be generated with the following lines of Python code:

```python
from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="<ROOT>", extra="<EXTRA>")
    dataset.dump_extra()
```

Note that the root and extra directories do not have to be distinct directories.

### ImageNet-21k

For Imagenet-21K, ensure your dataset is stored as shards, with the following directory structure
<pre>
Imagenet-21K
├── n01791314.tar
├── n02215621.tar
├── n02638596.tar
├── n02979290.tar
├── n03326073.tar
├── n03659292.tar
├── n04018667.tar
|── [...]
</pre>

The metadata files (similar to Imagenet-1K) can be generated with the following lines of Python code:

```python
import os
from typing import Optional
from dinov2.data.datasets.image_net_22k import ImageNet22k

# Add this method to your ImageNet22k class
def get_root(self, root: Optional[str] = None) -> str:
    """Return the specified root or the default root if None is provided."""
    return root if root is not None else self.root


ImageNet22k.get_root = get_root

root_path = "<path to IN-21K>/"
extra_path = "<path to IN-21K>/extra/"

# Create an instance without loading extra files
dataset = ImageNet22k(root=root_path, extra=extra_path, initialize_only=True)

dataset.dump_extra()
```

<br />

:warning: To execute the commands provided in the next sections for training and evaluation, the `dinov2` package should be included in the Python module search path, i.e. simply prefix the command to run with `PYTHONPATH=.`.


### LAION-600M

To install the LAION-600M (LAION-COCO) dataset, please do the following:

```bash
# Install img2dataset
pip install img2dataset

# Create dataset directory and move into it
mkdir -p laion-coco && cd laion-coco/

# Download all 128 parquet shards from Hugging Face
for i in {0..127}; do
    wget "https://huggingface.co/datasets/laion/laion-coco/resolve/main/part-$(printf "%05d" $i)-2256f782-126f-4dc6-b9c6-e6757637749d-c000.snappy.parquet"
done

# Move back to the root directory
cd ..

# Run img2dataset to download and preprocess the images
img2dataset \
  --url_list laion-coco \
  --input_format "parquet" \
  --url_col "URL" \
  --caption_col "TEXT" \
  --output_format webdataset \
  --output_folder laion-coco \
  --processes_count 16 \
  --thread_count 64 \
  --image_size 512 \
  --resize_only_if_bigger=True \
  --resize_mode="keep_ratio" \
  --skip_reencode=True \
  --save_additional_columns '["similarity","hash","punsafe","pwatermark","top_caption","all_captions","all_similarities"]' \
  --enable_wandb True
```

Unlike Imagenet-21K, we do not need any metadata files for LAION-600M.
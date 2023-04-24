# Robust Steganography without Embedding based on Secure Container Synthesis and Iterative Message Recovery (IJCAI 2022)

## Requirements
+ One high-end NVIDIA GPU with at least 11GB of memory. We have done all testing using a NVIDIA RTX 2080Ti.
+ Python >= 3.7 and PyTorch >= 1.8.2.
+ CUDA toolkit 11.2 or later.

## Quick Inference

### Preparation

1. Clone this repo:
```shell
git clone https://github.com/Lemok00/SI-SWE.git
cd SI-SWE
```
2. Install dependent packages:
```shell
pip install lmdb numpy opencv-python pandas tqdm clean-fid
```
3. Copy dataset statistics files to the directory of `clean-fid`:    
```shell
cp stats/* your_python_library/site-packages/cleanfid/stats/
# for example: cp stats/* ~/miniconda/envs/si-swe/lib/python3.8/lib/site-packages/cleanfid/stats/
```
or create your custom dataset statistics:
```python
from cleanfid import fid
fid.make_custom_stats(custom_name, dataset_path, mode="clean")
```

### Download checkpoints 

Our SI-SWE checkpoints can be found under following link: [Google Drive](https://drive.google.com/drive/folders/1r2RbDuVk1U3dZRROtZ5DIk0Llh5gUF1Y?usp=sharing)

Download checkpoints and organize the `./experiments/` folder as following structure.

    ./experments
        └── SI-SWE                  # model_name
            ├── 256-Bedroom         # experiment name
            │   ├── config.txt      # experiment configs
            │   └── model
            │       └── 500000.pt   # model checkpoint
            ├── 256-Church
            │   └── ... 
            └── ... 

### Evaluate message recovery accuracy under attacks

Ensure that the checkpoints with configs are already downloaded and origanized, then run the following command:

```shell
python test_acc_under_attacks.py --model model_name --exp_name exp_name --sigma sigma
# for example: python test_acc_under_attacks.py --model SI-SWE --exp_name 256-Bedroom --sigma 1
```
The result will be stored in `./results/robustness_evaluation/{model_name}/sigma={sigma}/{exp_name}.csv`.

### Evaluate message recovery accuracy with fake factors

Run the following command:
```shell
python test_acc_with_fake_factors.py --model model_name --exp_name exp_name
# for example: python test_acc_with_fake_factors.py --model SI-SWE --exp_name 256-Bedroom
```

### Evaluate Synthesis fidelity

Run the following command:
```shell
python synthesise_images_and_calculate_fid.py --model model_name --exp_name exp_name --synthesise_images --calculate_fid
# for example: python test_acc_with_fake_factors.py --model SI-SWE --exp_name 256-Bedroom
```
The synthesised images will be stored in `./results/synthesised_images/{model_name}/{exp_name}/`.
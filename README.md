# Robust Steganography without Embedding based on Secure Container Synthesis and Iterative Message Recovery (IJCAI 2022)

paper | supplementary material

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
    Or create your custom dataset statistics:
    ```shell
        from cleanfid import fid
        fid.make_custom_stats(custom_name, dataset_path, mode="clean")
    ```

### Download checkpoints 
    Our SI-SWE checkpoints can be found under following link: [Google Drive]
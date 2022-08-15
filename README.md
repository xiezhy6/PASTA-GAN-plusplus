# PASTA-GAN++: A Versatile Framework for High-Resolution Unpaired Virtual Try-on

Official implementation of "PASTA-GAN++: A Versatile Framework for High-Resolution Unpaired Virtual Try-on".

## Requirements

Create a virtual environment:
```
virtualenv pasta --python=3.7
source pasta/bin/activate
```
Install required packages:
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
pip install psutil scipy matplotlib opencv-python scikit-image==0.18.3 pycocotools
apt install libgl1-mesa-glx
```

## Running Inference
We provide the [pre-trained models](https://drive.google.com/file/d/1oESyGm1Zcz2lWUO6AvKlj-pXWtvIRGZd/view?usp=sharing) of PASTA-GAN++ which are trained by using the full UPT dataset (i.e., our newly collected data, data from Deepfashion dataset, data from MPV dataset) with the resolution of 512 separately.

we provide some test data under the directory `test_datas`, and provide a simple script to test the pre-trained model provided above on the UPT dataset as follow:
```
CUDA_VISIBLE_DEVICES=0 python3 -W ignore test.py \
    --dataroot test_datas --testtxt test_pairs.txt \
    --network checkpoints/pasta-gan++/network-snapshot-004408.pkl \
    --outdir test_results/upper \
    --batchsize 1 --testpart upper
```
or you can run the bash script by using the following command:
```
bash test.sh 1
```

Note that, in the testing script, the parameter `--network` refers to the path of the pre-trained model, the parameter `--outdir` refers to the path of the directory for generated results, the parameter `--dataroot` refers to the path of the data root, the parameter `--testtxt` refers to pair list of the garment-person pairs, the parameter `--testpart` refers to the garment part PASTA-GAN++ conducts the garment transfer. As for the configuration for these parameters, please refer to `test.sh`.
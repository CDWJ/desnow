# Data download
Run following commands to get training set, test set and realistic images.
```
wget https://desnownet.s3.amazonaws.com/dataset_synthetic/test/Snow100K-testset.tar.gz
wget https://desnownet.s3.amazonaws.com/dataset_synthetic/train/Snow100K-training.tar.gz
wget https://desnownet.s3.amazonaws.com/realistic_image/realistic.tar.gz
```
Then run the following command to unzip the file.
```
tar -xvzf Snow100K-training.tar.gz
tar -xvzf Snow100K-testset.tar.gz
tar -xvzf realistic.tar.gz
mv ./media/jdway/GameSSD/overlapping/test ./
```
# Conda env
Run
```
conda create -n desnow
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install ignite -c pytorch
conda install matplotlib
conda install numpy
conda install -c anaconda scikit-learn
conda install scikit-image
pip install visdom
pip install dominate
```

# Before training
Run following command to split train and validate set.
```
python split_script.py
```
To train CycleSnowGAN, please create a `trainA` and `trainB` folder under `./pytorch-CycleGAN-and-pix2pix/dataset` and copy the images from `./all/gt` to `trainA` and `./all/synthetic` to `trainB`

# Train CycleSnowGAN
```
cd pytorch-CycleGAN-and-pix2pix
python -m visdom.server -port port_number_for_visdom
```
If using UNet-256
```
python train.py --data_root ./dataset/snow --name your_exp_name --model cycle_gan --direction BtoA --display_port port_number_for_visdom --aspp True --netG unet_256
```
If using Resnet
```
python train.py --data_root ./dataset/snow --name your_exp_name --model cycle_gan --direction BtoA --display_port port_number_for_visdom --aspp True --netG resnet_9blocks
```
Other train/test flag information can be found in `options` folder

# Train DesnowNet
```
tensorboard --logdir ./  --port 6006  --host 0.0.0.0
```
Modify train/test options if needed in` train_v4downup.py` or `train.py`. The former one is for downsampling+upsampling. The later one is for keeping dim through all training time.
```
python train_v4downup.py
```

We also changed the crop size and learning rate. To change learning rate, see the training python script. To change crop size, or resizing images, see network script. Network without downsampling is in `network.py` and with downsampling is in `network_upsampling.py`.

# See Result
We provide `result.ipynb` to visualize result. See the notebook for more detail.


# SiamDCA
Here is the implementation of the proposed SiamDCA visual tracker, 
based on the Keras and Tensorflow deep learning API. 

The codes are able to run in Linux as well as Windows10 environment. 

Raw results can be downloaded from our 
[Google Drive](https://drive.google.com/drive/folders/1wY_Egkht6Yiib1Tg_ak_KERCz7m_Svlk?usp=sharing) 
or [Baidu NetDisk](https://pan.baidu.com/s/1FspS1YAC1q65uoUYI9rL2A) (extraction code: `sdca`).

## Installation

### Requirements
* Conda with Python 3.7
* Nvidia GPU
* CUDA Toolkit 10.0 with corresponding CUDNN
* Tensorflow-GPU 1.14.0
* Keras 2.3.1
* h5py 2.10.0
* OpenCV-python 
* matplotlib

### Instructions

#### Create environment and activate
```bash
conda create --name siamdca python=3.7
source activate siamdca
```

#### Install tensorflow/keras
```bash
pip install tensorflow-gpu=1.14.0 keras=2.3.1 h5py=2.10.0
```

#### Install other requirements
```bash
pip install opencv-python matplotlib tdqm Cython colorama future yacs
```

#### Build extensions
```bash
python setup.py build_ext --inplace
```

#### Add the tracker to PYTHONPATH
```bash
export PYTHONPATH=/path/to/SiamDCA/:$PYTHONPATH
```

### Preparation
Downloading models, datasets and corresponding jsons from our 
[Google Drive](https://drive.google.com/drive/folders/1wY_Egkht6Yiib1Tg_ak_KERCz7m_Svlk?usp=sharing) 
or [Baidu NetDisk](https://pan.baidu.com/s/1FspS1YAC1q65uoUYI9rL2A) (extraction code: `sdca`).

#### Download models
Put the pretrained backbone and weights of whole model in `weights` dir. 

#### Download datasets and jsons
In this work, we use the following training datasets:
* [COCO](http://cocodataset.org)
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [GOT10K](http://got-10k.aitestunion.com/)
* [LASOT](https://cis.temple.edu/lasot/)

Don't forget put the training json files of training data in `json_lables` dir.

Validating datasets:
- [VOT2016](http://www.votchallenge.net/vot2016/dataset.html)
- [VOT2018](http://www.votchallenge.net/vot2018/dataset.html)
- [VOT2019](http://www.votchallenge.net/vot2019/dataset.html)
- [VOT2018-LT](http://www.votchallenge.net/vot2018/dataset.html)
- [OTB100](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html)
- [UAV123](https://ivul.kaust.edu.sa/Pages/Dataset-UAV123.aspx)
- [NFS](http://ci2cv.net/nfs/index.html)

The json files of testing datasets should be put in the corresponding folders, like:
```
|-- UAV123
    |-- frame
    |--anno
    |-- data_seq
        |-- bike1
        |-- ...
        |-- wakeboard10
        |-- VOT2018.json 
```
You can also download them from [PySOT](https://github.com/STVIR/pysot).

Note that, unlike other SiamRPN-based trackers, 
in this work we directly use the vanilla images to build training pairs, 
instead of the raw pre-processed patches cropped from original images like PySOT.

For convenience, we put the datasets outside the project folder and use the absolute path in codes.
Thus, after downloading and unzipping the datasets, 
don't forget to set their path in [configs/DataPath.py](configs/DataPath.py)


## Demo
```bash
cd /path/to/SiamDCA/
conda activate siamdca
python demo.py 	\
	--tracker SiamDCA 	\ 
	--gpu_id 1          \ # GPU ID
	--track_config VOT2018.yaml     # config file
```

After building model and load weights, the program will request for the path of the demonstration video.

There are 3 demo modes in total:

(1)Path of the image sequence, like 
`F://DataBase/Benchmark/Basketball/img/`
or
`/home/DataBase/VOT2018/ants1/`

(2) Path of the video, whose format should end with '.mp4' or '.avi', like
`./test_videos/bag.avi`
or
`E://DataBase/MyVideos/202105151049.mp4`

(3)If having a webcam, 
input nothing, directly pressing the `ENTER` key on keyboard, to test the video stream from the webcam.

When running and showing,
you can click and select the window, and then press the `ESC` key to
break up the immediate test and choose the next one.

##  Train
```bash
python  train_SiamDCA.py	\
	--tracker SiamDCA 	\ # name of your tracker
	--log_dir SiamDCA     # where saving checkpoints
	--num_gpu 2         \ # number of GPUs
	--gpu_id 0, 1          \ # GPU ID
	--num_worker 16         
	--max_queue_size 32      
```

By default, the code will turn on the multiprocessing option in model.fit() (a class method of the Model class in Keras API),
and it can automatically determine whether the net gets distributed according to the number of available gpus.

## Test
We totally provide 5 test modes, including:

Test Mode 1: Evaluate the performance of a tracker with corresponding config on the chosen dataset
```bash
python  evaluate/test_DCA.py	\
	--tracker VOT2018 	\ # SiamDCA
	--gpu_id 1          \ # GPU ID
	--track_config VOT2018.yaml     # config file
```

Test Mode 2: Evaluate the performance of a tracker with corresponding config on all datasets

Test Mode 3: Visualized evaluation

Test Mode 4: Hyper Parameters search

Test Mode 5: Grid search

Read the code and decide the way of evaluation.


## Evaluate

``` bash
python evaluate/eval.py 	 \
	--dataset VOT2018        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix SiamDCA   # tracker name
```

## Structure

```
|-- SiamDCA
    |-- configs
    |   |-- DataPath.py         # absolute paths of both testing and training datasets
    |   |-- base_config.py      # all basic parameters of model and training
    |   |-- New_Tracker         # configs of a certain tracker
    |       |-- base.yaml       # model settings demand modifying, like num_filters
    |       |-- train_settings  # training settings, including augmenting, label assigning, loss weights...
    |-- json_labels             # jsons of training datasets
    |       |-- coco-train.yaml     
    |       |-- vid-val.yaml       
    |       |-- ...
    |-- weights                 # weights and pretrained model (.h5)
    |   |-- New_Tracker         
    |       |-- epoch65.h5      
    |       |-- ...         
    |   |-- EfficientNet-B3.h5         
    |   |-- ...  
    |-- experiments             # evaluation configs of a certain tracker
    |   |-- New_Tracker         
    |       |-- VOT2018.yaml       
    |       |-- UAV123.yaml
    |       |-- ...    
    |-- logs                    # training checkpoints (.h5)
    |   |-- New_Tracker_ver02
    |       |-- epoch15.h5      
    |       |-- epoch47.h5
    |       |-- ...
    |-- results                 # evaluating results (.txt)
    |   |-- VOT2016
    |       |-- SiamFC       
    |       |-- New_Tracker
    |       |-- ...
    |-- model                   
    |   |-- Backbone            # Add new backbones here
    |       |-- __init__.py     # set the output layers and filters, and the paths of pretrained weights    
    |       |-- EfficientNet.py
    |       |-- ...   
    |   |-- Neck                # Add new necks here
    |       |-- __init__.py     
    |       |-- FPN.py
    |       |-- ...   
    |   |-- Layers              # Add custom operations and layers here (Layer class in Keras API)
    |       |-- __init__.py 
    |       |-- activation.py   # deformable convolution, depth-wise correlation...
    |       |-- activation.py   # swish, mish...
    |       |-- operation.py    # split, transpose, stack...
    |       |-- ...
    |   |-- Model           # Add new models here
    |       |-- __init__.py 
    |       |-- New_Model.py    # Choose components and assemble them into a new Siamese Network
    |       |-- ...   
    |-- training                # loss functions, assign labels, read and augment data...
    |-- utils                   # iou, image (normalize, resize,...), anchor, box (clip, corner-to-center...), ...
    |-- evaluate         
```

## Design, train and test a new tracker
- Step 1: Build network Model.

Choose existing/design new operations and layers in `model` 
and assemble them into a novel Siamese network,
which is an instance of the Model class in Keras API. 
It accepts at least two inputs: X and Z, and output the classification maps, regression maps and so on.
Backbone, neck, head and other modules are the sub-models.
Then write the model settings in a yaml file and put it in `configs/New_Tracker`. 

- Step 2: Build training Model.

Denote the regression targets and classification labels as the Keras Input Layer.
Choose or design loss functions (packaged as Keras Layer in `training\Loss`) to obtain the losses between labels and predictions of network.
Then build the **train model** (also a Keras Model), which accepts image patches and labels at a same time and outputs multiple losses.

- Step 3: Write training script referring to `train_SiamDCA.py`. 

The data pipeline is based on Keras Sequence class.
Choose or design label assigner and box encoder in `training\BoxEncoder`

- Step 4: Training and evaluating.

Run training scripts. Choose a checkpoint in `logs/New_Tracker_ver2`.
Set tracking configs and put the yaml files in `experiments/New_Tracker`.
Then run `evaluate/test_DCA.py` to produce results txt files in `results` and calculate `evaluate/eval.py` to get final performance.


## License

This project is released under the [Apache 2.0 license](LICENSE). 

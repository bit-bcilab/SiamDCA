

import platform

SYSTEM = platform.system()

if SYSTEM == 'Linux':
    TEST_PATH = {'OTB50': '/home/Data/Benchmark/',
                 'OTB100': '/home/Data/Benchmark/',
                 'OTB-hard': '/home/Data/Benchmark/',
                 'LaSOT': '/home/Data/LaSOT/LaSOTBenchmark/',
                 'UAV': '/home/Data/UAV123/data_seq/UAV123/',
                 'GOT-10k': '/home/Data/GOT-10k/test/',
                 'TC128': '/home/Data/Temple-color-128/',
                 'VOT2016': '/home/Data/VOT2016/',
                 'VOT2018': '/home/Data/VOT2018/',
                 'VOT2019': '/home/Data/VOT2019/',
                 'NFS30': '/home/Data/NFS/',
                 'NFS240': '/home/Data/NFS/',
                 'VOT2018-LT': '/home/Data/vot2018_lt/',
                 'TrackingNet': '/home/Data/TrackingNet/'}
    ROOT_PATH = '/home/Data/training_dataset/'
    GOT_PATH = '/home/Data/GOT-10k/'
    LASOT_PATH = '/home/Data/LaSOT/LaSOTBenchmark/'

else:
    TEST_PATH = {'OTB50': 'F://DataBase/Benchmark/',
                 'OTB100': 'F://DataBase/Benchmark/',
                 'OTB-hard': 'F://DataBase/Benchmark/',
                 'LaSOT': 'I://LaSOT/LaSOTBenchmark/',
                 'UAV': 'F://DataBase/UAV123/Dataset_UAV123/UAV123/data_seq/UAV123/',
                 'GOT-10k': 'E://DataBase/GOT-10k/test/',
                 'TC128': 'F://DataBase/Temple-color-128/',
                 'VOT2016': 'F://DataBase/VOT2016/',
                 'VOT2018': 'F://DataBase/VOT2018/',
                 'VOT2019': 'F://DataBase/VOT2019/',
                 'NFS30': 'E://BaiduNetdiskDownload/Nfs/',
                 'NFS240': 'E://BaiduNetdiskDownload/Nfs/',
                 'VOT2018-LT': 'E://BaiduNetdiskDownload/vot2018_lt/',
                 'TrackingNet': 'E://BaiduNetdiskDownload/TN/'}
    ROOT_PATH = 'E://DataBase/training_dataset/'
    GOT_PATH = 'E://DataBase/GOT-10k/'
    LASOT_PATH = 'I://LaSOT/LaSOTBenchmark/'

COCO_PATH = 'coco/'
DET_PATH = 'det/ILSVRC/Data/DET/train/'
VID_PATH = 'vid/ILSVRC2015/Data/VID/train'

TRAIN_PATH = dict(COCO=ROOT_PATH + COCO_PATH, COCO_val=ROOT_PATH + COCO_PATH,
                  DET=ROOT_PATH + DET_PATH, DET_val=ROOT_PATH + DET_PATH,
                  VID=ROOT_PATH + VID_PATH, VID_val=ROOT_PATH + VID_PATH,
                  GOT=GOT_PATH + '/train', GOT_val=GOT_PATH + '/val',
                  LaSOT=LASOT_PATH)
TRAIN_JSON_PATH = 'json_labels/'



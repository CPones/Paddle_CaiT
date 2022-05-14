# 飞桨训推一体认证（TIPC）

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了PaddleOCR中所有模型的飞桨训推一体认证 (Training and Inference Pipeline Certification(TIPC)) 信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。

![tipc导图](./image/guide.png)

## 2. 测试工具简介
### 目录介绍

```shell
test_tipc/
├── configs/                        # 配置文件目录
    ├── CaiT                        # 2s-AGCN 模型的测试配置文件目录 
        ├── train_infer_python.txt  # 测试Linux上python训练预测（基础训练预测）的配置文件
├── results/  # 结果
├── output/   # 测试结果日志
├── prepare.sh                        # 完成test_*.sh运行所需要的数据和模型下载
├── test_train_inference_python.sh    # 测试python训练预测的主程序
└── readme.md                         # 使用文档
```

### 测试流程概述

使用本工具，可以测试不同功能的支持情况，以及预测结果是否对齐，测试流程概括如下：

1. 运行prepare.sh准备测试所需数据和模型；
2. 运行要测试的功能对应的测试脚本`test_train_inference_python.sh`，产出log，由log可以看到不同配置是否运行成功；

测试单项功能仅需两行命令，**如需测试不同模型/功能，替换配置文件即可**，命令格式如下：
```shell
# 功能：准备数据
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/prepare.sh  configs/[model_name]/[params_file_name]  [Mode]
# 功能：运行测试
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/test_train_inference_python.sh configs/[model_name]/[params_file_name]  [Mode]
```

以下为示例：
```shell
# 功能：准备数据
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/prepare.sh ./test_tipc/configs/CaiT/train_infer_python.txt 'lite_train_lite_infer'
# 功能：运行测试
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/CaiT/train_infer_python.txt 'lite_train_lite_infer'
```

## 3.TIPC测试成功日志
日志文件保存于`test_tipc/lite_train_lite_infer`
```
merging config from CaiT/configs/cait_xxs24_224.yaml
0513 11:06:40 PM config= AMP: False
BASE: ['']
DATA:
  BATCH_SIZE: 4
  BATCH_SIZE_EVAL: 8
  CROP_PCT: 1.0
  DATASET: imagenet2012
  DATA_PATH: data/LSVRC2012
  IMAGE_SIZE: 224
  NUM_WORKERS: 2
EVAL: False
LOCAL_RANK: 0
MODEL:
  ATTENTION_DROPOUT: 0.0
  DROPOUT: 0.1
  NAME: cait_xxs24_224
  NUM_CLASSES: 1000
  PRETRAINED: CaiT/cait_xxs24_224
  RESUME: None
  TRANS:
    DEPTH: 24
    DEPTH_TOKEN_ONLY: 2
    EMBED_DIM: 192
    INIT_VALUES: 1e-05
    IN_CHANNELS: 3
    MLP_RATIO: 4.0
    NUM_HEADS: 4
    PATCH_SIZE: 16
    QKV_BIAS: True
  TYPE: cait
NGPUS: -1
PREDICT: False
REPORT_FREQ: 100
SAVE: ./output/train-20220513-23-06-40
SAVE_FREQ: 5
SEED: 0
TAG: default
TRAIN:
  ACCUM_ITER: 2
  BASE_LR: 1e-05
  END_LR: 0.0
  GRAD_CLIP: 1.0
  LAST_EPOCH: 0
  LR_SCHEDULER:
    DECAY_EPOCHS: 30
    DECAY_RATE: 0.1
    MILESTONES: 30, 60, 90
    NAME: warmupcosine
  NUM_EPOCHS: 2
  OPTIMIZER:
    BETAS: (0.9, 0.999)
    EPS: 1e-08
    MOMENTUM: 0.9
    NAME: AdamW
  WARMUP_EPOCHS: 3
  WARMUP_START_LR: 0.0
  WEIGHT_DECAY: 0.05
VALIDATE_FREQ: 100
W0513 23:06:40.876610  4423 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
W0513 23:06:40.880825  4423 device_context.cc:465] device: 0, cuDNN Version: 7.6.
----- Imagenet2012 image train list len = 501
----- Imagenet2012 image val list len = 501
----- Imagenet2012 image test list len = 1
0513 11:06:44 PM ----- Pretrained: Load model state from CaiT/cait_xxs24_224
0513 11:06:44 PM Start training from epoch 1.
0513 11:06:44 PM Now training epoch 1. LR=0.000003
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:130: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:130: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
0513 11:06:44 PM Epoch[001/002], Step[0000/0126], Avg Loss: 0.7160, Avg Acc: 1.0000
0513 11:06:58 PM Epoch[001/002], Step[0100/0126], Avg Loss: 0.9230, Avg Acc: 0.7871
0513 11:07:02 PM ----- Epoch[001/002], Train Loss: 0.9668, Train Acc: 0.7804, time: 17.83
0513 11:07:02 PM Now training epoch 2. LR=0.000007
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:130: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:130: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
0513 11:07:02 PM Epoch[002/002], Step[0000/0126], Avg Loss: 1.7335, Avg Acc: 0.7500
0513 11:07:16 PM Epoch[002/002], Step[0100/0126], Avg Loss: 0.9182, Avg Acc: 0.7822
0513 11:07:19 PM ----- Epoch[002/002], Train Loss: 0.9611, Train Acc: 0.7725, time: 17.23
0513 11:07:19 PM ----- Validation after Epoch: 2
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:130: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:130: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
0513 11:07:19 PM Val Step[0000/0063], Avg Loss: 1.4068, Avg Acc@1: 0.6250, Avg Acc@5: 0.7500
0513 11:07:22 PM ----- Epoch[002/002], Validation Loss: 1.0431, Validation Acc@1: 0.7505, Validation Acc@5: 0.9162, time: 2.86
0513 11:07:22 PM ----- Save model: ./output/train-20220513-23-06-40/cait-Epoch-2-Loss-0.9611145798079744.pdparams
0513 11:07:22 PM ----- Save optim: ./output/train-20220513-23-06-40/cait-Epoch-2-Loss-0.9611145798079744.pdopt
 Run successfully with command - python CaiT/main_single_gpu.py -cfg='CaiT/configs/cait_xxs24_224.yaml' -dataset='imagenet2012' -batch_size=10 -pretrained='CaiT/cait_xxs24_224' -data_path='data/LSVRC2012' -output_dir=./log/CaiT/lite_train_lite_infer/norm_train_gpus_0 -epochs=2 -pretrained=CaiT/cait_xxs24_224 -batch_size=4!  
merging config from CaiT/configs/cait_xxs24_224.yaml
W0513 23:07:25.084822  4556 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
W0513 23:07:25.089610  4556 device_context.cc:465] device: 0, cuDNN Version: 7.6.
Model is saved in infer/best_model.
 Run successfully with command - python CaiT/export_model.py --model_path=output/best_model.pdparams --pretrained=./log/CaiT/lite_train_lite_infer/norm_train_gpus_0/latest.pdparams --save-inference-dir=./log/CaiT/lite_train_lite_infer/norm_train_gpus_0!  
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:130: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/auto_log/env.py:53: DeprecationWarning: distro.linux_distribution() is deprecated. It should only be used as a compatibility shim with Python's platform.linux_distribution(). Please use distro.id(), distro.version() and distro.name() instead.
  plat = distro.linux_distribution()[0]
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/auto_log/env.py:54: DeprecationWarning: distro.linux_distribution() is deprecated. It should only be used as a compatibility shim with Python's platform.linux_distribution(). Please use distro.id(), distro.version() and distro.name() instead.
  ver = distro.linux_distribution()[1]
[2022/05/13 23:07:42] root INFO: 

[2022/05/13 23:07:42] root INFO: ---------------------- Env info ----------------------
[2022/05/13 23:07:42] root INFO:  OS_version: Ubuntu 16.04
[2022/05/13 23:07:42] root INFO:  CUDA_version: 10.1.243
[2022/05/13 23:07:42] root INFO:  CUDNN_version: 7.3.1
[2022/05/13 23:07:42] root INFO:  drivier_version: 460.32.03
[2022/05/13 23:07:42] root INFO: ---------------------- Paddle info ----------------------
[2022/05/13 23:07:42] root INFO:  paddle_version: 2.2.2
[2022/05/13 23:07:42] root INFO:  paddle_commit: b031c389938bfa15e15bb20494c76f86289d77b0
[2022/05/13 23:07:42] root INFO:  log_api_version: 1.0
[2022/05/13 23:07:42] root INFO: ----------------------- Conf info -----------------------
[2022/05/13 23:07:42] root INFO:  runtime_device: gpu
[2022/05/13 23:07:42] root INFO:  ir_optim: True
[2022/05/13 23:07:42] root INFO:  enable_memory_optim: True
[2022/05/13 23:07:42] root INFO:  enable_tensorrt: False
[2022/05/13 23:07:42] root INFO:  enable_mkldnn: False
[2022/05/13 23:07:42] root INFO:  cpu_math_library_num_threads: 1
[2022/05/13 23:07:42] root INFO: ----------------------- Model info ----------------------
[2022/05/13 23:07:42] root INFO:  model_name: classification
[2022/05/13 23:07:42] root INFO:  precision: fp32
[2022/05/13 23:07:42] root INFO: ----------------------- Data info -----------------------
[2022/05/13 23:07:42] root INFO:  batch_size: 1
[2022/05/13 23:07:42] root INFO:  input_shape: dynamic
[2022/05/13 23:07:42] root INFO:  data_num: 1
[2022/05/13 23:07:42] root INFO: ----------------------- Perf info -----------------------
[2022/05/13 23:07:42] root INFO:  cpu_rss(MB): 2362.1016, gpu_rss(MB): 964.0, gpu_util: 4.0%
[2022/05/13 23:07:42] root INFO:  total time spent(s): 1.738
[2022/05/13 23:07:42] root INFO:  preprocess_time(ms): 1722.6572, inference_time(ms): 15.321, postprocess_time(ms): 0.0565
image_name: data/LSVRC2012/train/n02100877/n02100877_10277.JPEG, class_id: 213, prob: 10.112760543823242
 Run successfully with command - python CaiT/infer.py --model-dirs=CaiT/infer --img-path='data/LSVRC2012/train/n02100877/n02100877_10277.JPEG' --use-gpu=True --model-dir=./log/CaiT/lite_train_lite_infer/norm_train_gpus_0 --batch-size=1   --benchmark=True > ./log/CaiT/lite_train_lite_infer/python_infer_gpu_batchsize_1.log 2>&1 !  
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:130: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/auto_log/env.py:53: DeprecationWarning: distro.linux_distribution() is deprecated. It should only be used as a compatibility shim with Python's platform.linux_distribution(). Please use distro.id(), distro.version() and distro.name() instead.
  plat = distro.linux_distribution()[0]
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/auto_log/env.py:54: DeprecationWarning: distro.linux_distribution() is deprecated. It should only be used as a compatibility shim with Python's platform.linux_distribution(). Please use distro.id(), distro.version() and distro.name() instead.
  ver = distro.linux_distribution()[1]
[2022/05/13 23:07:50] root INFO: 

[2022/05/13 23:07:50] root INFO: ---------------------- Env info ----------------------
[2022/05/13 23:07:50] root INFO:  OS_version: Ubuntu 16.04
[2022/05/13 23:07:50] root INFO:  CUDA_version: 10.1.243
[2022/05/13 23:07:50] root INFO:  CUDNN_version: 7.3.1
[2022/05/13 23:07:50] root INFO:  drivier_version: 460.32.03
[2022/05/13 23:07:50] root INFO: ---------------------- Paddle info ----------------------
[2022/05/13 23:07:50] root INFO:  paddle_version: 2.2.2
[2022/05/13 23:07:50] root INFO:  paddle_commit: b031c389938bfa15e15bb20494c76f86289d77b0
[2022/05/13 23:07:50] root INFO:  log_api_version: 1.0
[2022/05/13 23:07:50] root INFO: ----------------------- Conf info -----------------------
[2022/05/13 23:07:50] root INFO:  runtime_device: cpu
[2022/05/13 23:07:50] root INFO:  ir_optim: True
[2022/05/13 23:07:50] root INFO:  enable_memory_optim: True
[2022/05/13 23:07:50] root INFO:  enable_tensorrt: False
[2022/05/13 23:07:50] root INFO:  enable_mkldnn: False
[2022/05/13 23:07:50] root INFO:  cpu_math_library_num_threads: 1
[2022/05/13 23:07:50] root INFO: ----------------------- Model info ----------------------
[2022/05/13 23:07:50] root INFO:  model_name: classification
[2022/05/13 23:07:50] root INFO:  precision: fp32
[2022/05/13 23:07:50] root INFO: ----------------------- Data info -----------------------
[2022/05/13 23:07:50] root INFO:  batch_size: 1
[2022/05/13 23:07:50] root INFO:  input_shape: dynamic
[2022/05/13 23:07:50] root INFO:  data_num: 1
[2022/05/13 23:07:50] root INFO: ----------------------- Perf info -----------------------
[2022/05/13 23:07:50] root INFO:  cpu_rss(MB): 1598.7656, gpu_rss(MB): None, gpu_util: None%
[2022/05/13 23:07:50] root INFO:  total time spent(s): 2.9778
[2022/05/13 23:07:50] root INFO:  preprocess_time(ms): 2800.4625, inference_time(ms): 177.2206, postprocess_time(ms): 0.0732
image_name: data/LSVRC2012/train/n02100877/n02100877_10277.JPEG, class_id: 213, prob: 10.11275863647461
 Run successfully with command - python CaiT/infer.py --model-dirs=CaiT/infer --img-path='data/LSVRC2012/train/n02100877/n02100877_10277.JPEG' --use-gpu=False --model-dir=./log/CaiT/lite_train_lite_infer/norm_train_gpus_0 --batch-size=1   --benchmark=True > ./log/CaiT/lite_train_lite_infer/python_infer_cpu_batchsize_1.log 2>&1 !  
```

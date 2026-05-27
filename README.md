# RetinaFace-MobileNet0.25 Face Detection

本项目基于 RetinaFace 人脸检测框架进行整理与二次开发，主要用于 **RetinaFace-MobileNet0.25 轻量化人脸检测模型的训练、测试与 ONNX 推理部署**。当前版本主要围绕 MobileNet0.25 backbone 展开，适合用于人脸检测、五点人脸关键点定位应用场景。

> 说明：本仓库代码参考并整理自 RetinaFace 相关开源实现，当前主要用于学习、实验复现和工程化开发。项目中重点使用的是 RetinaFace-MobileNet0.25 预训练权重，其他预训练权重暂未在本仓库中整理说明。

---

## 1. 项目简介

RetinaFace 是一种单阶段人脸检测方法，能够在一次前向推理中同时输出人脸检测框和五点人脸关键点。本项目采用轻量化的 **MobileNet0.25** 作为 backbone，相比 ResNet 系列模型更加适合在资源受限环境中进行推理和部署。

当前项目重点包括：

- RetinaFace-MobileNet0.25 人脸检测模型训练；
- 人脸框与五点关键点检测；
- 模型测试与推理；
- ONNX 模型导出；
- ONNX Runtime 推理测试；

---

## 2. 当前支持情况

当前版本主要支持 **RetinaFace-MobileNet0.25**。

| 模型 | Backbone | 当前状态 |
|---|---|---|
| RetinaFace-MobileNet0.25 | MobileNet0.25 | 已用于训练、测试和 ONNX 导出 |
| RetinaFace-ResNet50 | ResNet50 | 暂未整理 |
| RetinaFace-ResNet152 | ResNet152 | 暂未整理 |

本仓库 README 仅说明当前实际使用的 MobileNet0.25 版本，其他预训练权重暂不展开。

---

## 3. 数据组织方式

1.本项目训练所需的注释文件和 RetinaFace-MobileNet0.25 预训练权重已通过百度网盘分享。链接: https://pan.baidu.com/s/1SC00qfK2Nc9zOE_yq6Gv9g?pwd=7ehb

2.下载数据集[WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html)

3.建议将数据集放置在项目根目录下的 `data/retinaface/` 中，

```text
Retinaface/
├── data/
│   └── retinaface/
│       ├── train/
│       │   ├── images/
│       │   └── label.txt
│       ├── val/
│       │   ├── images/
│       │   └── label.txt
│       └── test/
│           ├── images/
│           └── label.txt
├── model/
│   ├──mobilenet_0_25-0000.params
    └──mobilenet_0_25-symbol.json
├── rcnn/
├── train.py
├── test.py
├── test_widerface.py
├── export_retinaface_onnx.py
├── retinaface_onnx_infer_test.py
└── README.md
```

如果实际数据路径不同，需要同步修改 `rcnn/config.py` 或相关脚本中的数据路径。

---

## 4. 环境配置

建议使用 Conda 创建独立环境：

```bash
conda create -n retinaface python=3.8 -y
conda activate retinaface
```

安装基础依赖：

```bash
pip install numpy opencv-python tqdm easydict cython
pip install onnx onnxruntime
```

本项目原始训练流程依赖 MXNet。请根据自己的 CUDA 版本安装对应版本的 MXNet。

如果使用 GPU，可安装对应 CUDA 版本的 MXNet，例如：

```bash
pip install mxnet-cu112
```

如果仅使用 CPU，可安装：

```bash
pip install mxnet
```

安装完成后，编译 C++/Cython 工具：

```bash
make
```

如果 `make` 报错，请检查系统是否安装了以下工具：

```text
gcc
g++
make
cython
```

---

## 5. 配置文件准备

训练前需要将示例配置文件复制为正式配置文件：

```bash
cp rcnn/sample_config.py rcnn/config.py
```

然后检查并修改：

```text
rcnn/config.py
```
本项目主要使用轻量化 MobileNet0.25 配置，对应训练命令中的参数为：

```bash
--network mnet
```

---

## 6. 训练

使用 RetinaFace-MobileNet0.25 进行训练：

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --prefix ./model/retina --network mnet
```

参数说明：

- `CUDA_VISIBLE_DEVICES='0,1,2,3'`：指定使用第 0、1、2、3 张 GPU；
- `python -u train.py`：启动训练脚本；
- `--prefix ./model/retina`：指定模型保存前缀；
- `--network mnet`：使用 MobileNet0.25 轻量化网络配置。

如果只使用单张 GPU，可以改为：

```bash
CUDA_VISIBLE_DEVICES='0' python -u train.py --prefix ./model/retina --network mnet
```
---

## 7. 测试

模型测试可参考项目中的测试脚本：

```bash
python test.py
```

或者使用 WiderFace 测试脚本：

```bash
python test_widerface.py
```
---

## 8. ONNX 模型导出

本项目整理了 RetinaFace-MobileNet0.25 的 ONNX 导出脚本，可执行：

```bash
python export_retinaface_onnx.py
```

导出后可得到 ONNX 模型文件，用于后续部署或跨框架推理。

---

## 9. ONNX 推理测试

本项目提供 ONNX 推理测试脚本，可运行：

```bash
python retinaface_onnx_infer_test.py
```

---

## 10. 致谢

本项目参考 RetinaFace 相关开源实现，并在其基础上针对 RetinaFace-MobileNet0.25 训练、测试和 ONNX 推理流程进行了整理。感谢原作者在人脸检测和人脸关键点定位方面的开源工作。

RetinaFace 论文引用如下：

```bibtex
@inproceedings{Deng2020CVPR,
  title = {RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild},
  author = {Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
  booktitle = {CVPR},
  year = {2020}
}
```

---

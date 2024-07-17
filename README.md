**MICCAI 2023 Semi-supervised Teeth Segmentation Challenge**

**竞赛成果：**录用MACCAI workshop 论文一篇

其他成员：Li Zheng

**赛题背景：**

随着医疗技术的不断进步，医学图像处理领域引起了越来越多的关注，尤其是在口腔医学成像这一关键领域。作为牙科成像的核心，牙齿图像分割在疾病检测、性别确定和身份识别等应用中发挥着关键作用。牙齿图像分割的目标是精确识别和隔离感兴趣的区域，为牙医提供可靠的诊断基础。然而，牙齿复杂的解剖结构，包括牙釉质、牙本质和牙髓等多种成分，构成了显著的挑战。这些成分之间的模糊边界使图像分割任务变得非常复杂。此外，口腔环境充满唾液和反射等众多干扰元素，进一步降低了牙齿图像的质量，增加了分割的难度。

**数据来源：**“阿里天池MICCAI 2023 Challenges：STS-基于2D全景图像的牙齿分割任务”

**方案：**

BFFNet主要由编码网络、边界特征提取模块和特征交叉融合模块组成。图 1 描述了我们的牙齿图像分 割模型的总体框架。接下来的部分对整体架构和模型的关键要素进行了深入研究。面对牙齿周围神经组织的复杂性以及由此产生的模糊边界分割问题，本研究提出了 BFFNet，一种创新的边界特征融合网络。该网络的设计核心分为两部分：边界特征提取模块和特征交叉融合模块。

**算法结构图：**

![image](https://github.com/zfgTOfruirer/Tooth_Seg/tree/master/images/fig-1.png)

**BFFNet**:**在PraNet的基础上，改进添加BFEM模块与FCFM模块。使算法提取全局特征的同时重点关注边界特征。**

**FCFM:特征交叉融合模块**

![image](.\images\fig-3.png)



**BFEM:边界特征提取模块**

![image](.\images\fig-2.png)

**算法代码：**

lib：模型定义模块以及相关依赖

--- BFFNet\_Model

--- ResNet\_ALL

utils：预处理脚本

--- dataloader

--- format\_conversion

BFFNet\_Test.py：测试

BFFNet\_Train.py：训练

README.md：项目说明




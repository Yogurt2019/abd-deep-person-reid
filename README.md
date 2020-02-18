# ABD-Net on Torch-Reid

ABD-Net在Torch-Reid(latest 1.0.9)上的实现。

###使用方法

在工程根目录下输入：

    python setup.py develop
    
编译工程。
    
将ABD-Net文件夹放入deep-person-reid的projects文件夹中，将abd_resnet放入torchreid/models文件夹中，在

    torchreid/models/__init__.py
    
中加入：

    from .abd_resnet import *

在

    __model_factory

中加入：

    'abd_resnet': ABD_resnet50,

即成功导入带有ABD分支的ResNet50模型。

具体用法和Deep-Person-Reid相同。

###引用

[Deep-Person-Reid](https://github.com/KaiyangZhou/deep-person-reid)

[ABD-Net](https://github.com/TAMU-VITA/ABD-Net)


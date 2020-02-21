import os
import torch.nn as nn


# 获取最后一级文件夹的路径
def get_path(root):
    """
    :param root: 根目录的路径
    :return: 最后一级文件夹的所有路径
    """
    path_list = []

    def _get_path(root_path):
        if os.path.isfile(root_path) or not os.listdir(root_path):
            if os.path.isfile(root_path):
                path_list.append(os.path.dirname(root_path))
            else:
                path_list.append(root_path)
        else:
            for dir in os.listdir(root_path):
                _get_path(os.path.join(root_path, dir))

    _get_path(root)
    return list(set(path_list))


def init_params(x):

    if x is None:
        return

    for m in x.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.normal_(m.weight, 1, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def open_specified_layers(model, open_layers):
    """
    Open specified layers in model for training while keeping
    other layers frozen.

    Args:
    - model (nn.Module): neural net model.
    - open_layers (list): list of layer names.
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    # for layer in open_layers:
    #     assert hasattr(model, layer), "'{}' is not an attribute of the model, please provide the correct name".format(layer)

    for name, module in model.named_children():
        # if name in open_layers:
        if name in open_layers:
            print('open', name)
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
            open_specified_layers(module, open_layers)
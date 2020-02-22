import torch
import torchvision.models as models
from torchreid.models.abd_resnet import ABD_resnet50
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import json
import xlwt
from projects.ABD_Net.ABD_components.tools.utils import get_path
import pickle
import time

path_list = pickle.load(open('/media/ddj2/8b1bfd93-3f3f-4475-b279-6a9ae59c6639/remote_dir/abd-deep-person-reid/path_list.pkl', 'rb'))
label_dict = pickle.load(open('/media/ddj2/8b1bfd93-3f3f-4475-b279-6a9ae59c6639/results/abd_resnet/label_dict.pkl', 'rb'))


def get_hierarchy(label):
    global path_list
    hierarchy = []
    path = None
    for item in path_list:
        if item.split('/')[-1] == label:
            path = item
            break
    if path is None:
        raise FileNotFoundError('No existing path')
    per_path_list = path.split('/')
    crop_index = per_path_list.index('crop') + 1
    for item in per_path_list[crop_index:]:
        hierarchy.append(item)
    return hierarchy


def predict():
    device = "cuda:0,1,2,3" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dir = '/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/V20200108/zhejiang_train_torch_resnet50/train'

    class_names = list(label_dict.keys())
    val_dir = '/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/测试集/ceshi/crop/浙江省温州苍南县西古庵早白垩世小平田组PM201(挑选3张泛化测试用）20200114'
    # val_dir = '/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/V20200108/zhejiang_train_torch_resnet50/val'
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    topk = 3

    checkpoint = torch.load('/media/ddj2/8b1bfd93-3f3f-4475-b279-6a9ae59c6639/remote_dir/abd-deep-person-reid/projects/ABD_Net/log/resnet50_abd/model.pth.tar-40')
    model = ABD_resnet50(num_classes=70)
    model = torch.nn.DataParallel(model).cuda()
    time.sleep(0.1)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    f = xlwt.Workbook()
    sheet1 = f.add_sheet('浙江top3', cell_overwrite_ok=True)
    sheet2 = f.add_sheet('浙江top1', cell_overwrite_ok=True)

    first_line = ['文件夹', '文件', '信心值', '第1级标签', '第2级标签', '第3级标签', '第4级标签', '第5级标签', '第6级标签', '第7级标签', '第8级标签']
    for i in range(len(first_line)):
        sheet1.write(0, i, first_line[i])
        sheet2.write(0, i, first_line[i])
    line = 1
    line2 = 1
    l = 0

    with torch.no_grad():
        for epoch_idx, (inputs, _) in enumerate(val_loader):
            inputs = inputs.cuda()
            outputs = model(inputs)
            softmax = torch.nn.Softmax(dim=1)
            outputs = softmax(outputs)
            preds = torch.argsort(outputs).cpu().numpy()[:, ::-1][:, :topk]
            # batch
            for batch_id in range(inputs.size()[0]):
                pred_label = []
                for j in range(topk):
                    pred_label.append(preds[batch_id][j])
                # topk
                hierarchies = []
                for label in pred_label:
                    hierarchy = get_hierarchy(class_names[label])
                    hierarchies.append(hierarchy)
                # orig folder
                sheet1.write_merge(line, line + topk - 1, 0, 0, val_dataset.imgs[l][0].rsplit('/')[-2])
                sheet2.write(line2, 0, val_dataset.imgs[l][0].rsplit('/')[-2])
                # orig file
                sheet1.write_merge(line, line + topk - 1, 1, 1, val_dataset.imgs[l][0].rsplit('/')[-1])
                sheet2.write(line2, 1, val_dataset.imgs[l][0].rsplit('/')[-1])

                for i in range(topk):
                    for j in range(len(hierarchies[i])):
                        sheet1.write(line, 2, str(outputs[batch_id][preds[batch_id][i]].cpu().numpy().tolist()))
                        sheet1.write(line, j + 3, hierarchies[i][j])
                    line += 1
                for j in range(len(hierarchies[0])):
                    sheet2.write(line2, 2, str(outputs[batch_id][preds[batch_id][0]].cpu().numpy().tolist()))
                    sheet2.write(line2, j + 3, hierarchies[0][j])
                line2 += 1
                print('Pred', batch_id, ' ',
                      val_dataset.imgs[l][0].rsplit('/')[-2], '/', val_dataset.imgs[l][0].rsplit('/')[-1],
                      '==> ', class_names[preds[0][0]])
                l += 1
        f.save('abd_resnet_result.xls')


def main():
    predict()


if __name__ == '__main__':
    main()

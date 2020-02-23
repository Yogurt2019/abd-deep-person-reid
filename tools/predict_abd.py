import torch
from torchreid.models.abd_resnet import ABD_resnet50
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import xlwt
import pickle
import time
import os.path as osp
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
from torchreid.metrics.distance import compute_distance_matrix
from torch.nn import functional as F

path_list = pickle.load(
    open('/media/ddj2/8b1bfd93-3f3f-4475-b279-6a9ae59c6639/remote_dir/abd-deep-person-reid/path_list.pkl', 'rb'))


class CompareImage:
    def __init__(self, info):
        self.img_size = 672
        self.img = None
        self.info = info
        # info: [dataset_root, save_dir, orig_file, orig_class, top1, top1_dist, top2, top2_dist]
        self.text_pos = [self.img_size + 5, self.img_size + 5, self.img_size + 5, self.img_size + 5, self.img_size + 5]
        self.create_img()

    def add_text(self, text, pos, text_color=(255, 255, 255), text_size=20):
        assert isinstance(text, str), "text must be a string"
        draw = ImageDraw.Draw(self.img)
        fontstyle = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                                       text_size,
                                       encoding='utf-8')
        x = self.img_size * pos
        y = self.text_pos[pos]
        position = (x, y)
        if len(text) > 34:
            draw.text(position, text[:34], text_color, font=fontstyle)
            self.text_pos[pos] += 25
            x = self.img_size * pos
            y = self.text_pos[pos]
            position = (x, y)
            draw.text(position, text[34:], text_color, font=fontstyle)
            self.text_pos[pos] += 25
        else:
            draw.text(position, text, text_color, font=fontstyle)
            self.text_pos[pos] += 25

    def L2(self, yhat, y):
        loss = np.sum(np.power((y - yhat), 2))
        return loss

    def choose_img(self, origin_img, path):
        min_loss = float('inf')
        ret_img = None
        for img in os.listdir(path):
            img_arr = cv2.imread(osp.join(path, img))
            cur_loss = self.L2(cv2.imread(origin_img), img_arr)
            if cur_loss < min_loss:
                ret_img = img
        return osp.join(path, ret_img)

    def get_hierarchy(self, label):
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
        ret = ''
        for item in hierarchy:
            ret += item + '-'
        ret = ret[:-1]
        return ret

    def get_whole_hierarchy(self, file_path) -> str:
        p_list = file_path.split('/')
        p_list = p_list[p_list.index('crop'):-2]
        ret = ''
        for item in p_list:
            ret += item + '/'
        ret = ret[:-1]
        return ret

    def get_true_img(self, orig_img, orig_class):
        global path_list
        orig_path = None
        for path in path_list:
            if orig_class in path:
                orig_path = path
                break
        if orig_path is None:
            raise ValueError("path not in path_list, can't find path in dataset")
        true_img = self.choose_img(orig_img, orig_path)
        return true_img

    def get_img_pos(self, cur_pos):
        axis = (self.img_size * cur_pos, 0, self.img_size * (cur_pos + 1), self.img_size)
        return axis

    def plot_heatmap(self, heatmap_feature, img):
        heatmap_feature = torch.unsqueeze(heatmap_feature, dim=0)
        outputs = (heatmap_feature ** 2).sum(1)
        b, h, w = outputs.size()
        outputs = outputs.view(b, h * w)
        outputs = F.normalize(outputs, p=2, dim=1)
        outputs = outputs.view(b, h, w)

        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]

        for t, m, s in zip(img, img_mean, img_std):
            t.mul_(s).add_(m).clamp_(0, 1)
        img_np = np.uint8(np.floor(img.numpy() * 255))
        img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

        # activation map
        am = outputs.numpy()
        am = np.squeeze(am, axis=0)
        am = cv2.resize(am, (self.img_size, self.img_size))
        am = 255 * (am - np.min(am)) / (
                np.max(am) - np.min(am) + 1e-12
        )
        am = np.uint8(np.floor(am))
        am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

        # overlapped
        overlapped = img_np * 0.6 + am * 0.4
        overlapped[overlapped > 255] = 255
        overlapped = overlapped.astype(np.uint8)

        htmap_pil = Image.fromarray(overlapped)
        return htmap_pil

    def create_img(self):
        # info: [dataset_root, save_dir, orig_file, orig_class, top1, top1_dist, top2, top2_dist, hm_ft, ims]
        self.img = Image.new('RGB', (self.img_size * 5, 900))
        # 原图
        orig_img_pil = Image.open(self.info[2])
        cur_pos = 0
        self.img.paste(orig_img_pil, self.get_img_pos(cur_pos))
        self.add_text('原图：', cur_pos)
        self.add_text(self.get_hierarchy(self.info[2].split('/')[-2]), cur_pos)

        # 真值
        cur_pos += 1
        true_img = Image.open(self.get_true_img(self.info[2], self.info[3]))
        self.img.paste(true_img, self.get_img_pos(cur_pos))
        self.add_text('真值：', 1)
        self.add_text(self.get_hierarchy(self.info[2].split('/')[-2]), cur_pos)

        # top1
        cur_pos += 1
        top1_pil = Image.open(self.info[4])
        self.img.paste(top1_pil, self.get_img_pos(cur_pos))
        self.add_text('top 1:', cur_pos)
        self.add_text(self.get_hierarchy(self.info[4].split('/')[-2]), cur_pos)
        self.add_text('欧氏距离：', cur_pos)
        self.add_text(str(self.info[5]), cur_pos)

        # top2
        cur_pos += 1
        top2_pil = Image.open(self.info[6])
        self.img.paste(top2_pil, self.get_img_pos(cur_pos))
        self.add_text('top 2:', cur_pos)
        self.add_text(self.get_hierarchy(self.info[6].split('/')[-2]), cur_pos)
        self.add_text('欧氏距离：', cur_pos)
        self.add_text(str(self.info[7]), cur_pos)

        # heatmap
        cur_pos += 1
        hm_ft = self.info[8]
        overlapped_hm_pil = self.plot_heatmap(hm_ft, self.info[9])
        self.img.paste(overlapped_hm_pil, self.get_img_pos(cur_pos))
        self.add_text('模型热力图', cur_pos)

        # save img
        save_dir = osp.join(self.info[1], self.info[3])
        if not osp.exists(save_dir):
            os.mkdir(save_dir)
        self.img.save(osp.join(save_dir, self.info[2].split('/')[-1]))


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


def feature_extraction(data_loader, model, mode='gallery'):
    f, pids, heatmap_features, ims = [], [], [], []
    for batch_idx, data in enumerate(data_loader):
        if mode == 'gallery':
            imgs, pids_ = data[0], data[1]
        elif mode == 'query':
            imgs = data[0]
            for i in range(imgs.shape[0]):
                ims.append(imgs[i].cpu())
        else:
            raise ValueError("Mode must be one of (gallery, query)")
        imgs = imgs.cuda()
        outputs = model(imgs)
        features = outputs[0]
        if mode == 'query':
            heatmap_feature = outputs[3]['after'][0].data.cpu()
            for j in range(heatmap_feature.size(0)):
                heatmap_features.append(heatmap_feature[j, ...])
        features = features.data.cpu()
        f.append(features)
        if mode == 'gallery':
            pids.extend(pids_)
    f = torch.cat(f, 0)
    if mode == 'gallery':
        pids = np.asarray(pids)
        return f, pids
    else:
        return f, heatmap_features, ims


def predict():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    gallery_dir = '/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/V20200108/zhejiang_train/rock_dataset/gallery'
    gallery_dataset = datasets.ImageFolder(
        gallery_dir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        num_workers=32
    )

    val_dir = '/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/测试集/ceshi/crop/浙江省温州苍南县西古庵早白垩世小平田组PM201(挑选3张泛化测试用）20200114'
    # val_dir = '/media/ddj2/8b1bfd93-3f3f-4475-b279-6a9ae59c6639/test'
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        num_workers=32
    )

    checkpoint = torch.load(
        '/media/ddj2/8b1bfd93-3f3f-4475-b279-6a9ae59c6639/remote_dir/abd-deep-person-reid/projects/ABD_Net/log/resnet50_abd/model.pth.tar-40')
    model = ABD_resnet50(num_classes=70)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # set model to eval mode before extracting features
    print('Extracting features from query set ...')
    qf, heatmap_features, ims = feature_extraction(val_loader, model, mode='query')
    print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

    print('Extracting features from gallery set ...')
    gf, pids = feature_extraction(gallery_loader, model, mode='gallery')
    print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    distmat = compute_distance_matrix(qf, gf)
    distmat = distmat.numpy()

    indices = np.argsort(distmat, axis=1)

    topk = 3

    f = xlwt.Workbook()
    sheet1 = f.add_sheet('浙江top3', cell_overwrite_ok=True)
    sheet2 = f.add_sheet('浙江top1', cell_overwrite_ok=True)

    first_line = ['文件夹', '文件', '欧氏距离', '第1级标签', '第2级标签', '第3级标签', '第4级标签', '第5级标签', '第6级标签', '第7级标签', '第8级标签']
    for i in range(len(first_line)):
        sheet1.write(0, i, first_line[i])
        sheet2.write(0, i, first_line[i])
    line = 1
    line2 = 1
    # l indicates the position of dataset
    l = 0

    for batch_id in range(indices.shape[0]):
        pred_label = []
        for j in range(topk):
            pred_label.append(indices[batch_id][j])
        # topk
        hierarchies = []
        for label in pred_label:
            hierarchy = get_hierarchy(gallery_dataset.imgs[label][0].split('/')[-2])
            hierarchies.append(hierarchy)
        # orig folder
        orig_class = val_dataset.imgs[l][0].rsplit('/')[-2]
        sheet1.write_merge(line, line + topk - 1, 0, 0, orig_class)
        sheet2.write(line2, 0, val_dataset.imgs[l][0].rsplit('/')[-2])
        # orig file
        orig_file = val_dataset.imgs[l][0].rsplit('/')[-1]
        sheet1.write_merge(line, line + topk - 1, 1, 1, orig_file)
        sheet2.write(line2, 1, val_dataset.imgs[l][0].rsplit('/')[-1])

        for i in range(topk):
            for j in range(len(hierarchies[i])):
                sheet1.write(line, 2, str(distmat[batch_id][i].tolist()))
                sheet1.write(line, j + 3, hierarchies[i][j])
            line += 1
        for j in range(len(hierarchies[0])):
            sheet2.write(line2, 2, str(distmat[batch_id][0].tolist()))
            sheet2.write(line2, j + 3, hierarchies[0][j])
        line2 += 1
        top1_dist = distmat[batch_id][indices[batch_id][0]]
        top2_dist = distmat[batch_id][indices[batch_id][1]]
        top1 = gallery_dataset.imgs[pred_label[0]][0]
        top2 = gallery_dataset.imgs[pred_label[1]][0]
        print('Pred', batch_id, ' ',
              val_dataset.imgs[l][0].rsplit('/')[-2], '/', val_dataset.imgs[l][0].rsplit('/')[-1],
              '==> ', gallery_dataset.imgs[pred_label[0]][0].split('/')[-2])
        l += 1

        # Plot Settings
        orig_file = val_dataset.imgs[l][0]
        dataset_root = '/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/V20200108/crop'
        save_dir = '/media/ddj2/8b1bfd93-3f3f-4475-b279-6a9ae59c6639/results/abd_resnet/plot'
        info = [dataset_root, save_dir, orig_file, orig_class, top1, top1_dist, top2, top2_dist,
                heatmap_features[batch_id], ims[batch_id]]
        save = CompareImage(info)

    f.save('abd_resnet_result.xls')


def main():
    predict()


if __name__ == '__main__':
    main()

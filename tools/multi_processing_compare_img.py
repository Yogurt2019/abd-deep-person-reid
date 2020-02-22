import xlrd
import xlwt
import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool
from tqdm import tqdm
from PIL.ExifTags import TAGS
import exifread


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


def get_exif(test_img):
    origin_test_img = test_img.replace('crop', 'images')
    origin_test_img_pil = open(origin_test_img, 'rb')
    exif = exifread.process_file(origin_test_img_pil)
    return exif


def L2(yhat, y):
    loss = np.sum(np.power((y-yhat), 2))
    return loss


def choose_img(path, origin_img, dest_imgs):
    min_loss = float('inf')
    ret_img = None
    for img in dest_imgs:
        img_arr = cv2.imread(os.path.join(path, img))
        cur_loss = L2(cv2.imread(origin_img), img_arr)
        if cur_loss < min_loss:
            ret_img = img
    return ret_img


def add_text(img, text, pos, text_color=(255, 255, 255), text_size=20):
    draw = ImageDraw.Draw(img)
    fontstyle = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                                   text_size,
                                   encoding='utf-8')
    draw.text(pos, text, text_color, font=fontstyle)
    return img


def compare(process_num, start_num):
    #test_root = '/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/测试集/ceshi/crop/福建火山岩--寄到北京的泛化标本的识别测试照片（精挑2块按目录分开全）'
    test_root = '/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/测试集/ceshi/crop/浙江省温州苍南县西古庵早白垩世小平田组PM201(挑选3张泛化测试用）20200114'
    #test_root = '/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/测试集/ceshi/crop/安徽第四批泛化独立测试样本（简3）'
    dataset_root = '/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/V20200108/crop'
    dataset_paths = get_path(dataset_root)

    xls_file = '/media/ddj2/8b1bfd93-3f3f-4475-b279-6a9ae59c6639/remote_dir/pytorch_image_classification/result.xls'
    wb = xlrd.open_workbook(xls_file)
    # 福建2浙江4安徽6
    sheet = 0
    cur_sheet = wb.sheet_by_index(sheet)
    s = int(cur_sheet.nrows/3/process_num*start_num)*3+1
    t = int((cur_sheet.nrows-1)/3/process_num*(start_num+1))*3
    print(s, '/', t)
    for row in range(s, t, 3):
        test_img = os.path.join(test_root, cur_sheet.cell(row, 0).value)
        test_img = os.path.join(test_img, cur_sheet.cell(row, 1).value)
        should_be_img_path = None
        for path in dataset_paths:
            p = cur_sheet.cell(row, 0).value
            if sheet == 2:
                should_be_img_path = p
            else:
                if p in path:
                    should_be_img_path = path
        if sheet != 2:
            should_be_imgs = os.listdir(should_be_img_path)
            should_be_img = os.path.join(should_be_img_path, choose_img(should_be_img_path, test_img, should_be_imgs))
        predict_img_path = dataset_root
        top2 = ''
        top2_predict_img_path = dataset_root
        for col in range(3, 10):
            predict_img_path = os.path.join(predict_img_path, cur_sheet.cell(row, col).value)
            top2_predict_img_path = os.path.join(top2_predict_img_path, cur_sheet.cell(row+1, col).value)
            add = cur_sheet.cell(row+1, col).value
            top2 += add
            if add:
                top2 += '/'
        top2 = top2[:-1]
        top1_confidence = cur_sheet.cell(row, 2).value
        top2_confidence = cur_sheet.cell(row+1, 2).value
        top2_predict_imgs = os.listdir(top2_predict_img_path)
        top2_predict_img = os.path.join(top2_predict_img_path, choose_img(top2_predict_img_path, test_img, top2_predict_imgs))
        predict_imgs = os.listdir(predict_img_path)
        predict_img = os.path.join(predict_img_path, choose_img(predict_img_path, test_img, predict_imgs))
        # target_img = np.zeros((900, 2018, 3), dtype='float32')
        # 672 1344 2016 2688
        if sheet == 2:
            target_img = Image.new('RGB', (2016, 905))
        else:
            target_img = Image.new('RGB', (2688, 905))
        test_img_pil = Image.open(test_img)
        if sheet != 2:
            should_be_img_pil = Image.open(should_be_img)
        predict_img_pil = Image.open(predict_img)
        top2_predict_img_pil = Image.open(top2_predict_img)
        # add text to image
        test_img_class = test_img[test_img.index('crop')+5:test_img.index('/')]
        predict_img_class = predict_img_path[predict_img_path.index('crop')+5:]
        if sheet != 2:
            should_be_img_class = should_be_img_path[should_be_img_path.index('crop') + 5:]
        else:
            should_be_img_class = should_be_img_path

        target_img.paste(test_img_pil, (0, 0, 672, 672))
        if sheet != 2:
            target_img.paste(should_be_img_pil, (672, 0, 1344, 672))
            target_img.paste(predict_img_pil, (1344, 0, 2016, 672))
            target_img.paste(top2_predict_img_pil, (2016, 0, 2688, 672))
        else:
            target_img.paste(predict_img_pil, (672, 0, 1344, 672))
            target_img.paste(top2_predict_img_pil, (1344, 0, 2016, 672))

        LINE_MAX_LEN = 35
        # 原图
        target_img = add_text(target_img, '原图', (5, 675))
        exif = get_exif(test_img)
        target_img = add_text(target_img, '品牌:'+str(exif['Image Make'].printable), (5, 700))
        target_img = add_text(target_img, '型号:'+str(exif['Image Model'].printable), (5, 725))
        target_img = add_text(target_img, '曝光时间:'+str(exif['EXIF ExposureTime'].printable), (5, 750))
        target_img = add_text(target_img, '光圈值:'+'f/'+str(exif['EXIF FNumber'].values[0].num/exif['EXIF FNumber'].values[0].den), (5, 775))
        target_img = add_text(target_img, '焦距'+ str(exif['EXIF FocalLength'].values[0].num / exif['EXIF FocalLength'].values[0].den) + 'mm', (5, 800))
        if sheet != 2:
            exif = get_exif(should_be_img)
            # 非福建
            target_img = add_text(target_img, '真值', (672, 675))
            if len(should_be_img_class) > LINE_MAX_LEN:
                target_img = add_text(target_img, should_be_img_class[:LINE_MAX_LEN], (672, 700))
                target_img = add_text(target_img, should_be_img_class[LINE_MAX_LEN:], (672, 725))
            else:
                target_img = add_text(target_img, should_be_img_class, (672, 700))
            target_img = add_text(target_img, '品牌:' + str(exif['Image Make'].printable), (672, 750))
            target_img = add_text(target_img, '型号:' + str(exif['Image Model'].printable), (672, 775))
            target_img = add_text(target_img,
                                  '曝光时间:' + str(exif['EXIF ExposureTime'].printable), (672, 800))
            target_img = add_text(target_img, '光圈值:' + 'f/' + str(exif['EXIF FNumber'].values[0].num/exif['EXIF FNumber'].values[0].den), (672, 825))
            target_img = add_text(target_img, '焦距' + str(exif['EXIF FocalLength'].values[0].num / exif['EXIF FocalLength'].values[0].den) + 'mm', (672, 850))
        if sheet != 2:
            start_point = 2
        else:
            start_point = 1
        target_img = add_text(target_img, '预测top1', (672*start_point, 675))
        if len(predict_img_class) > LINE_MAX_LEN:
            target_img = add_text(target_img, predict_img_class[:LINE_MAX_LEN], (672*start_point, 700))
            target_img = add_text(target_img, predict_img_class[LINE_MAX_LEN:-1], (672*start_point, 725))
        else:
            target_img = add_text(target_img, predict_img_class[:-1], (672*start_point, 700))
        target_img = add_text(target_img, 'top1信心:'+str(top1_confidence), (672*start_point, 750))
        exif = get_exif(predict_img)
        target_img = add_text(target_img, '品牌:' + str(exif['Image Make'].printable), (672*start_point, 775))
        target_img = add_text(target_img, '型号:' + str(exif['Image Model'].printable), (672*start_point, 800))
        target_img = add_text(target_img,
                              '曝光时间:' + str(exif['EXIF ExposureTime'].printable), (672*start_point, 825))
        target_img = add_text(target_img, '光圈值:' + 'f/' + str(exif['EXIF FNumber'].values[0].num/exif['EXIF FNumber'].values[0].den), (672*start_point, 850))
        target_img = add_text(target_img, '焦距' + str(exif['EXIF FocalLength'].values[0].num / exif['EXIF FocalLength'].values[0].den) + 'mm', (672*start_point, 875))
        start_point += 1
        target_img = add_text(target_img, '预测top2', (672*start_point, 675))
        if len(top2) > LINE_MAX_LEN:
            target_img = add_text(target_img, top2[:LINE_MAX_LEN], (672*start_point, 700))
            target_img = add_text(target_img, top2[LINE_MAX_LEN:], (672*start_point, 725))
        else:
            target_img = add_text(target_img, top2, (672 * start_point, 700))
        target_img = add_text(target_img, 'top2信心:'+str(top2_confidence), (672*start_point, 750))
        exif = get_exif(top2_predict_img)
        target_img = add_text(target_img, '品牌:' + str(exif['Image Make'].printable), (672 * start_point, 775))
        target_img = add_text(target_img, '型号:' + str(exif['Image Model'].printable), (672 * start_point, 800))
        target_img = add_text(target_img,
                              '曝光时间:' + str(exif['EXIF ExposureTime'].printable),
                              (672 * start_point, 825))
        target_img = add_text(target_img, '光圈值:' + 'f/' + str(exif['EXIF FNumber'].values[0].num/exif['EXIF FNumber'].values[0].den),
                              (672 * start_point, 850))
        target_img = add_text(target_img, '焦距' + str(exif['EXIF FocalLength'].values[0].num / exif['EXIF FocalLength'].values[0].den) + 'mm',
                              (672 * start_point, 875))

        save_dir = cur_sheet.cell(row, 0).value
        save_name = cur_sheet.cell(row, 1).value

        save_dir = os.path.join(os.path.join('/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/compare/浙江样本对比', save_dir))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # target_img.save(os.path.join('/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/compare/福建样本对比', str(row+1) + '.jpg'))
        #target_img.save(os.path.join('/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/compare/安徽样本对比', str(row+1) + '.jpg'))
        target_img.save(os.path.join(save_dir, save_name))


def main():
    process_num = 6
    po = Pool(process_num)
    for i in range(process_num):
        po.apply_async(compare, (process_num, i))
    po.close()
    po.join()
    # compare(1, 0)


if __name__ == '__main__':
    main()
# /media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/V20200108/crop/火山熔岩类/碎斑熔岩类/浅灰色酸性霏细状碎斑熔岩(福建南阳晚侏罗世南园组碎斑熔岩段D2605）{mlvJ3n}/IMG_20190523_083651.jpg

import os
import shutil


def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def main():
    dir = '/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/V20200108/zhejiang_onehot'
    ratio = [4, 4, 2]
    sum = 0
    for num in ratio:
        sum += num
    dirs = os.listdir(dir)
    for item in dirs:
        files = os.listdir(os.path.join(dir, item))
        for mode in ('train', 'gallery', 'query'):
            t_dir = os.path.join(dir.replace('zhejiang_onehot', 'zhejiang_train_8_2'), mode, item)
            makedir(t_dir)
            divide_point = []
            divide_point.append(int(len(files)*0.4))
            divide_point.append(divide_point[0] + int(len(files)*0.4))
            if mode == 'train':
                for file in files[:divide_point[0]]:
                    print('src:', os.path.join(dir, item, file))
                    print('dst:', t_dir)
                    shutil.copy(src=os.path.join(dir, item, file), dst=t_dir)
                    print()
            elif mode == 'gallery':
                for file in files[divide_point[0]:divide_point[1]]:
                    shutil.copy(src=os.path.join(dir, item, file), dst=t_dir)
                    print()
            else:
                for file in files[divide_point[1]:]:
                    shutil.copy(src=os.path.join(dir, item, file), dst=t_dir)
                    print()


if __name__ == '__main__':
    main()

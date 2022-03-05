import cv2, argparse, os
from glob import glob
from tqdm import tqdm

def parse_args():
    desc = "get the mean values of  b,g,r on the whole dataset"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='Hayao', help='dataset_name')

    return parser.parse_args()

def read_img(image_path):
    img = cv2.imread(image_path)
    assert len(img.shape) == 3
    B = img[..., 0].mean()
    G = img[..., 1].mean()
    R = img[..., 2].mean()
    return B,G,R

def get_mean(dataset_name):
    # file_list = glob(os.path.dirname(os.path.dirname(__file__))+'/../dataset/{}/{}/*.*'.format(dataset_name, 'style'))
    file_list = glob('/Users/fangzhengjie/Master/MyGAN/MyGAN/dataset/{}/{}/*.*'.format(dataset_name, 'style'))
    print(file_list)
    # os.path.dirname(__file__)返回脚本的路径
    # glob.glob函数的参数是字符串。这个字符串的书写和我们使用linux的shell命令相似，或者说基本一样。也就是说，只要我们按照平常使用cd命令时的参数就能够找到我们所需要的文件的路径。
    # 字符串中可以包括“*”、“?”和"["、"]"，其中“*”表示匹配任意字符串，“?”匹配任意单个字符，[0-9]与[a-z]表示匹配0-9的单个数字与a-z的单个字符
    image_num = len(file_list)
    print('image_num:', image_num)

    B_total = 0
    G_total = 0
    R_total = 0
    for f in tqdm(file_list) :
        bgr = read_img(f)
        B_total += bgr[0]
        G_total += bgr[1]
        R_total += bgr[2]

    B_mean, G_mean, R_mean = B_total / image_num, G_total / image_num, R_total / image_num
    mean = (B_mean + G_mean + R_mean)/3

    return mean-B_mean, mean-G_mean, mean-R_mean

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    return get_mean(args.dataset)

if __name__ == '__main__':
    result = main()
    print('style_data_mean_diff (B, G, R):  ', result)


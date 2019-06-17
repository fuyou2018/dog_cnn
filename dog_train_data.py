#coding=utf-8
 
import  os
import  numpy as np
import  cv2

dogs_path = './data/dogs'
nodog_path = './data/nodog'

size=64

imgs=[]
labs=[]

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)
    
    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def readData(path , h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            
            img = cv2.imread(filename)
            
            top,bottom,left,right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))
            
            imgs.append(img)
            labs.append(path)

readData(dogs_path)
readData(nodog_path)

images = np.array(imgs)
#labels = np.array(labs)
labels = np.array([[0,1] if lab == dogs_path else [1,0] for lab in labs])
#重新打乱
permutation = np.random.permutation(labels.shape[0])
all_images = images[permutation,:]
all_labels = labels[permutation,:]

#训练集与测试集比例 8：2
train_total = all_images.shape[0]
train_nums= int(all_images.shape[0]*0.8)
test_nums = all_images.shape[0]-train_nums

images = all_images[0:train_nums,:]
labels = all_labels[0:train_nums,:]


test_images = all_images[train_nums:train_total,:]
test_labels = all_labels[train_nums:train_total,:]



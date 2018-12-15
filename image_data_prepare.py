import scipy.io#用于加载mat文件
import numpy as np
import os
from PIL import Image
import shutil

labels = scipy.io.loadmat('I:\\dataSet\\imagelabels.mat')
labels = np.array(labels['labels'][0])-1


print("labels:",labels)

######## flower dataset: train test valid 数据id标识 ########
setid = scipy.io.loadmat('I:\\dataSet\\setid.mat')

validation = np.array(setid['valid'][0]) - 1
np.random.shuffle(validation)

train = np.array(setid['trnid'][0]) - 1
np.random.shuffle(train)

test=np.array(setid['tstid'][0]) -1
np.random.shuffle(test)
######## flower data path 数据保存路径 ########
flower_dir = list()

######## flower data dirs 生成保存数据的绝对路径和名称 ########
for img in os.listdir("I:\\dataSet\\102flowers"):
    
    ######## flower data ########
    flower_dir.append(os.path.join("I:\\dataSet\\102flowers", img))

######## flower data dirs sort 数据的绝对路径和名称排序 从小到大 ########
flower_dir.sort()

#print(flower_dir)

des_folder_train="I:\\dataSet\\prepare_pic\\train"
for tid in train:
    ######## open image and get label ########
    img=Image.open(flower_dir[tid])
    #print(flower_dir[tid])
    img = img.resize((256, 256),Image.ANTIALIAS)
    lable=labels[tid]
    #print(lable)
    
    path=flower_dir[tid]
    print("path:",path)
    
    base_path=os.path.basename(path)
    print("base_path:",base_path) 
    classes="c"+str(lable)
    class_path=os.path.join(des_folder_train,classes)
    # 判断结果
    if not os.path.exists(class_path):

        os.makedirs(class_path) 
    print("class_path:",class_path) 
    despath=os.path.join(class_path,base_path)
    print("despath:",despath)
    img.save(despath)
des_folder_validation="I:\\dataSet\\prepare_pic\\validation"

for tid in validation:
    ######## open image and get label ########
    img=Image.open(flower_dir[tid])
    #print(flower_dir[tid])
    img = img.resize((256, 256),Image.ANTIALIAS)
    lable=labels[tid]
    #print(lable)
    path=flower_dir[tid]
    print("path:",path)
    base_path=os.path.basename(path)
    print("base_path:",base_path) 
    classes="c"+str(lable)
    class_path=os.path.join(des_folder_validation,classes)
    # 判断结果
    if not os.path.exists(class_path):

        os.makedirs(class_path) 
    print("class_path:",class_path) 
    despath=os.path.join(class_path,base_path)
    print("despath:",despath)
    img.save(despath)
    
des_folder_test="I:\\dataSet\\prepare_pic\\test"

for tid in test:
    ######## open image and get label ########
    img=Image.open(flower_dir[tid])
    #print(flower_dir[tid])
    img = img.resize((256, 256),Image.ANTIALIAS)
    lable=labels[tid]
    #print(lable)
    path=flower_dir[tid]
    print("path:",path)
    base_path=os.path.basename(path)
    print("base_path:",base_path) 
    classes="c"+str(lable)
    class_path=os.path.join(des_folder_test,classes)
    # 判断结果
    if not os.path.exists(class_path):
        os.makedirs(class_path) 
    print("class_path:",class_path) 
    despath=os.path.join(class_path,base_path)
    print("despath:",despath)
    img.save(despath)
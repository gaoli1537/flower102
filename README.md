# 基于tensorflow_slim模型调参的flower102鲜花分类过程
    
 **小组成员:高立群(18029032) 吴庆(18029031) 赵晓娟(18026016)**   
    
    实验软件环境如下
    windows10
    tensorflow-gpu 1.11
    python3.5

## 1.数据分析工作
### 1.1数据介绍
实验所使用数据集由102类产自英国的花卉组成。每类由40-258张图片组成。具体示例如下图所示:
![花类别](https://my-picture-bed-1256685253.cos.ap-shanghai.myqcloud.com/201812/%E8%8A%B1%E7%A4%BA%E4%BE%8B.PNG)
下载地址为:

http://www.robots.ox.ac.uk/~vgg/data/flowers/102/

其中有两个mat文件标记了整个数据集的label,具体结构如下:

-imagelabels.mat
    
    总共有8189列，每列上的数字代表类别号。

-setid.mat    

    -trnid字段:总共有1020列，每10列为一类花卉的图片，每列上的数字代表图片号。    
    -valid字段:总共有1020列，每10列为一类花卉的图片，每列上的数字代表图片号。    
    -tstid字段:总共有6149列，每一类花卉的列数不定，每列上的数字代表图片号。
## 2.数据预处理

tensorflow-slim 程序包是由谷歌公司提供的图像分类工具包,其中预训练的比较流行的图像分类的神经网络,比如VGG16,VGG19,InceptionV1~V4,残差网络等等,实验中我们使用了比较新的InceptionV3模型进行训练.
### 2.1数据集图像格式处理

对于InceptionV3网络,要求输入的图片分辨率保持一致,由于数据集中的图片大小不一,所以需要修改分辨率后保存,这里将图片统一保存为256*256的jpg格式,具体代码如下:

```python
    #flower_dir[tid]为原图片的绝对地址
    img=Image.open(flower_dir[tid])
    img = img.resize((256, 256),Image.ANTIALIAS)
    #despath为生成标准图片的保存地址
    img.save(despath)
```
### 2.2数据集存储路径处理

在slim框架中,对于数据集的存储路径以及存储格式是由要求的,具体示例如下:
```
data_prepare/
    pic/
        train/
            class1/
                img1
                img2
                ...
            class2
                img1
                img2
                ...
        validation/
            class1/
                img1
                img2
                ...
            class2
                img1
                img2
                ...
```
所以需要根据数据集提供的标签规整图片的路径.总体代码如下:

```python
import scipy.io
import numpy as np
import os
from PIL import Image
import shutil

########取出 imagelabels 文件的值############

imagelabels_path='I:\\dataSet\\imagelabels.mat'
labels = scipy.io.loadmat(imagelabels_path)
labels = np.array(labels['labels'][0])-1

######## 取出 flower dataset: train test valid 数据id标识 ########
setid_path='I:\\dataSet\\setid.mat'
setid = scipy.io.loadmat(setid_path)

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

#####生成flower data train的分类数据 #######
des_folder_train="I:\\dataSet\\prepare_pic\\train"
for tid in train:
    ######## open image and get label ########
    img=Image.open(flower_dir[tid])
    #print(flower_dir[tid])
    ######## resize img #######
    img = img.resize((256, 256),Image.ANTIALIAS)
    lable=labels[tid]
    #print(lable)
    
    path=flower_dir[tid]
    #print("path:",path)
    
    base_path=os.path.basename(path)
    #print("base_path:",base_path) 
    ######类别目录路径
    classes="c"+str(lable)
    class_path=os.path.join(des_folder_train,classes)
    
    if not os.path.exists(class_path):
        os.makedirs(class_path) 
    
    #print("class_path:",class_path) 
    despath=os.path.join(class_path,base_path)
    #print("despath:",despath)
    img.save(despath)


#####生成flower data validation的分类数据 #######   
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


#####生成flower data test的分类数据 #######     
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
```
数据生成之后,共生成三个目录,分别为train,test,validation如下目录格式:

![训练集目录](https://my-picture-bed-1256685253.cos.ap-shanghai.myqcloud.com/201812/1.PNG)

![train目录格式示例](https://my-picture-bed-1256685253.cos.ap-shanghai.myqcloud.com/201812/3.PNG)

文件数量如下所示:
```
train:
    102类:1020个图片
validation:
    102类:1020幅图片
test:
    102类:6149幅图片
```

标准图片已经路径的处理工作完成之后,需要使用slim提供的脚本将图片转换为tfrecord格式,该格式作为tensorflow高速读取的二进制文件,数据的高速传输提供了接口,具体使用的教程可以[参考该博主](https://www.jianshu.com/p/78467f297ab5).

在实验过程中,我们使用预先编译好的脚本文件data_convert.py对图片进行转换,进入到该文件所在目录,使用如下命令:
``` 
    python data_convert.py -t I:\\prepare_data\\prepare_pic #生成图片根目录路径
    --train-shards 5\ #切成5两个tfrecord train文件
    --validation-shards 5\ #切成5两个tfrecord train文件
    --num-threads 5\  #启动五个线程运算
    --dataset-name flower102 #文件名头
```

运行完成后生成以下文件:

![tfrecord格式](https://my-picture-bed-1256685253.cos.ap-shanghai.myqcloud.com/201812/5.PNG)

## 3.模型选择

## 4.模型微调
### 4.1 拷贝文件到数据集目录

* 首先将生成的tfrecord文件以及label.txt拷贝到slim模型中,具体路径为slim/flower102/data

### 4.2定义新的datasets文件

对模型有一定的了解之后,我们进入到模型微调阶段,要将slim/datasets文件中的flowers.py做一些修改,并且另存flowers102.py具体修改以及解释如下:
```python
#将tfrecord文件的文件头改为flower102,对应生成tfrecord文件过程中的--dataset-name flower102命令
_FILE_PATTERN = 'flower102_%s_*.tfrecord'
# 设置训练集与验证集的图片个数,都是1020
SPLITS_TO_SIZES = {'train': 1020, 'validation': 1020}
#设置类别个数:102
_NUM_CLASSES = 102
#将图片格式改为"jpg"
keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }
```

修改完flowers102.py后,还需要对同目录下的dataset_factory.py进行修改,具体修改内容如下:

```python
from datasets import flower102
datasets_map = {

    'flower102':flower102,
}
```
具体就是把刚才新建的flower102添加到包中.

## 5.训练模型

### 5.1 准备训练文件夹:

在slim文件中建立以下目录结构:
```
slim/
    flower102/
        data/
        pretrained/
        train_dir/
```

* data中存放tfrecord数据,已经在4.1步完成

* pretrained中放置已经训练好的InceptionV3的模型,可以在网上下载,源文件中也已经包含.

* train_dir是用来保存训练过程中存储的模型的文件夹.

# 5.2 开始训练模型

在slim文件夹中,使用train_image_classifier.py文件对模型进行训练,具体命令行以及解释如下:
```
python train_image_classifier.py \
#模型保存路径
--train_dir=flower102/train_dir \
#数据集名称
--dataset_name=flower102 \
#数据集切分后的第二名称(train)
--dataset_split_name=train \
#数据集所在目录
--dataset_dir=flower102/data \
#使用的模型名称
--model_name=inception_v3 \
#使用的模型的地址
--checkpoint_path=flower102/pretrained/inception_v3.ckpt \
#微调层(在恢复训练模型时,不恢复这两层,这两层对V3模型的末端层,原模型对应1000类,而新模型只对应102类)
--checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
--trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
#最大迭代次数
--max_number_of_steps=100 \
#batch_size
--batch_size=32 
#学习率
--learning_rate=0.001 \
#学习率是否自动下降 此处为固定值
--learning_rate_decay_type=fixed \
#间隔多久保存一次模型
--save_interval_secs=50 \
#间隔多久写入日志以供tensorborad查看
--save_summaries_secs=2 \
#间隔迭代次数打印
--log_every_n_steps=10 \
#选定优化器
--optimizer=rmsprop \
#选定模型中2次正则化超参数
--weight_decay=0.00004 \
```
使用该命令对模型进行训练,训练过程部分截图如下:

![训练过程](https://my-picture-bed-1256685253.cos.ap-shanghai.myqcloud.com/201812/%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B.PNG)

使用tensorboard工具可以查看到损失函数下降的过程:
```python
tensorboard --logdir flower102/train_dir
```
![损失函数](https://my-picture-bed-1256685253.cos.ap-shanghai.myqcloud.com/201812/%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.PNG)

## 6.验证模型

验证过程与训练过程所使用的命令类似,如下:
```
python eval_image_classifier.py \
--checkpoint_path=/tmp/tfmodel/model.ckpt-10000 \
--eval_dir=flower102/eval_dir \
--dataset_dir=flower102/data \
--dataset_name=flower102 \
--dataset_split_name=validation \
--model_name=inception_v3
```
验证结果如下:

![验证结果](https://my-picture-bed-1256685253.cos.ap-shanghai.myqcloud.com/201812/%E9%AA%8C%E8%AF%81.PNG)

可以看出,准确率有83%,而top2的召回率有90%左右的成绩.

也可以使用tensorboard查看验证过程:

![结果](https://my-picture-bed-1256685253.cos.ap-shanghai.myqcloud.com/201812/%E7%BB%93%E6%9E%9C%E6%95%B0%E5%80%BC.PNG)

## 7.测试模型

### 7.1导出模型

tensorflow_slim提供了导出模型框架的脚本export_inference_graph.py,可以将模型框架导出,在通过使用freeze_graph.py将训练好的参数值导入到模型中去.

#### step 1

**输出框架**
```
python export_inference_graph.py \
--alsologtosterr \
--model_name=inception_v3 \
--output_file=flower102/inception_v3_inf_graph.pb \
--dataset_name flower102
```
#### step 2
**注入参数数据**

进入freeze_graph.py所在文件目录,输入:

```
python freeze_graph.py \
--input_graph slim/flower102/inception_v3_inf_graph.pb \
--input_checkpoint flower102\train_dir/model.ckpt-100000 \
--input_binary true \
--output_node_names InceptionV3/Predictions/Reshape_1 \
--output_graph slim/flower102/frozen_graph.pb
```

![验证模型生成](https://my-picture-bed-1256685253.cos.ap-shanghai.myqcloud.com/201812/%E7%94%9F%E6%88%90%E6%B5%8B%E8%AF%95%E6%A8%A1%E5%9E%8B.PNG)

经过这两步之后,带有参数值的模型就构造好了,接下来就可以使用这个模型进行测试工作:
运行根目录下的classify_image_incepetion_v3.py,并对以下输入参数进行修改,更正为自己所使用的测试图片与模型名称:

```python
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_path',#模型的路径,使用填充数据的模型框架
      default='slim/flower102/frozen_graph.pb',
      type=str,
  )
  parser.add_argument(
      '--label_path',#label地址,在生成tfrecord文件过程中自动生成了label.txt,制定为其地址.
      default='slim/flower102/data/label.txt',
      type=str,
  )
  parser.add_argument(
      '--image_file',#测试图片的地址,这里使用了相对地址
      type=str,
      default='image_07111.jpg',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',#给出top n的可能结果
      type=int,
      default=5,
      help='Display this many predictions.'
  )
```
以下为验证的结果截图:

测试image_07111.jpg这张图片,结果如下:

![png](https://my-picture-bed-1256685253.cos.ap-shanghai.myqcloud.com/201812/%E6%B5%8B%E8%AF%95.PNG)

可以看出C9的概率最高,对比该图片与C9类,可见结果正确.

![测试结果](https://my-picture-bed-1256685253.cos.ap-shanghai.myqcloud.com/201812/%E9%AA%8C%E8%AF%81%E7%BB%93%E6%9E%9C.PNG)


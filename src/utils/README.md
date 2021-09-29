# preprocess_img.py

该函数是进行原图片的处理，将所需要的图片部分截下来，保存到另一个文件夹中
修改代码中的root为train以及test存放的父目录，层级结构如下：
root
> train

>> trainset1

>> trainset2

>> trainset3

>> ……

> test

>> testset1

>> testset2
    
 运行代码 python preprocess_img.py 
 即可在pro_train，pro_val中获取图片
 
 
 # mix_images.py
 
 该函数是进行图片的融合，将上述函数处理好的结果——pro_train,pro_val放到文件夹split中
 运行代码 python mix_images.py
 即可获得融合后的图片

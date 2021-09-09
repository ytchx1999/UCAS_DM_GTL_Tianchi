# UCAS_DM_GTL_Tianchi
[[2021 亚太眼科学会大数据竞赛](https://tianchi.aliyun.com/competition/entrance/531929/information)] | [队伍：UCAS_DM_GTL]

```bash
.
├── LICENSE
├── README.md
├── TrainingAnnotation.csv
├── data_train.ipynb
├── src
└── tree.txt
```

---

First communication：

图片处理：（huanglinyan）

1. 将所需图片进行裁剪，获取有用的特征（同一个病例左右眼分开，治疗前后分开，命名样例：0000-0000L_1000_cut_1.jpg、0000-0000L_1000_cut_2.jpg，注意一下在遍历文件夹时，需要注意匹配前面的字段，后面的lr可能匹配不上)
2. 将裁剪下来的图片合并为一张图片（同一个病例左右眼分开，治疗前后分开，分别相加），图片保存到当前目录下，命名样例：0000-0000L_1.jpg 或 0000-0000L_2.jpg（1代表治疗前，2代表治疗后）

csv文件处理：（chihuixuan）

1. 去掉nan值所在的一个样本
2. 获取训练测试的输入数据以及label，一行样本对应处理后的一个文件名，治疗前后分开，行数扩大为原来的两倍（训练测试分别做一个txt文件）

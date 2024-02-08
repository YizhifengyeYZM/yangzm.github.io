# PyTorch 框架学习笔记

通常而言，可以将一个完整的 PyTorch 算法大框架分为下面几个部分：

- 最外层的主函数：
  - 读取配置文件，设置各项参数
  - 日志
  - 训练 / 测试流程
- Dataset 和 Data Loader 部分
  - Dataset 部分
  - Data Loader 部分
- Runner 或者 Schedule 部分
- 核心：网络部分



## Dataset 和 DataLoader 部分

> 这两部分负责将数据整理好，具体而言：
>
> - Dataset 类需要自定义好三个函数（`__init__(self)`，`__len__(self)`，`__getitem__(self, index)`），这个类的核心任务就是创建适应当前模型的数据集接口
> - DataLoader 则相对容易，基本上只要前面这个 Dataset 类搞好之后，构建 DataLoader 是非常容易的

### Dataset 部分

#### 1. Dataset 类模板

对于 2D Vision 而言，基本的 Dataset 类模板框架如下：

```py
class CashDataset(Dataset):
    # 1. 构造函数
    def __init__(self, data_dir, transform=None):
        self.data_path = """ 写一个从数据集目录'data_dir'中收集目标图像地址的函数"""
        self.transform = transform    # 对样本(图像/文本)进行处理的函数，最好是外面写好再输入进来
 	# 2. 获取单一数据样本的函数
    def __getitem__(self, index):
        path_img (,label) = self.data_path[index]	# 获取单一图像（或者其他对应的 label）
        img = Image.open(path_img).convert('RGB')   # 用 PIL 库打开图像  
 		if self.transform is not None:
            img = self.transform(img)   # 对输入图像进行处理（处理完之后必须是 tensor 类型）
		return img(, label)	            # 返回图像（和对应的其他标签等东西）
 	# 3. 获取整个数据集长度的函数
    def __len__(self):
        return len(self.data_path)

```

这个框架中比较重要的两点：

- 要从给定的整个数据集目录中整合出样本目录以及每个样本相对应的标签目录，这个函数要自己写
- 对于数据集中的每一个数据样本，是否要对其进行数据增强（Data Augmentation）？【训练时和测试时的处理肯定是不一样的！】如果要进行数据增强，则需要借助 PyTorch 中的`torchvision.transforms`【2D Vision，如果是其他任务可能得找别的处理方法】



#### 2. torchvision.transform 模块笔记

对于 2D Vision 任务而言，其实可以将许多图像增强的模块都整合到这个`torchvision.transform` 模块里面【连 Resize，Crop 这些都是可以的】

> 对于 torchvision.transform 模块而言，需要我们先用 `PIL.Image` 库打开图像



##### 2.1 容器类 `transforms.Compose`

首先需要知道的是容器类 `transforms.Compose`，这个类的任务只有一个：将一堆 transform 方法整合到一起称为一个 Pipeline，例如：

```py
from PIL import Image
from torchvision import transforms
img = Image.open("img 图片的路径")	# 先用 PIL 库打开图像

# transforms.Compose([...])：将其他图像处理方法全部整合到里面【用列表包装】
transformer = transforms.Compose([                                
    transforms.Resize(256),
    transforms.transforms.RandomResizedCrop((224), scale = (0.5,1.0)),	
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()	# 不管前面怎么操作，最后要把图像转换为 Tensor
])
# 应用模块的方法：直接套用
img_transformed = transformer(img)
```



##### 2.2 各种图像变换方法总结

`torchvision.transform` 中常见的图像变换方法：

- **对图像进行裁剪（Crop）：**
  - :star:中心裁剪：                          `transforms.CenterCrop`
  - :star:随机裁剪：                          `transforms.RandomCrop`
  - 随机长宽比裁剪：                  `transforms.RandomResizedCrop`
  - 上下左右中心裁剪：              `transforms.FiveCrop`
  - 上下左右中心裁剪后翻转：`transforms.TenCrop`
- **对图像进行翻转和旋转（Flip and Rotation）：**
  - :star:以概率 $p$ 水平翻转：`transforms.RandomHorizontalFlip(p=0.5)`
  - :star:以概率 $p$ 垂直翻转：`transforms.RandomVerticalFlip(p=0.5)`
  - :star:随机旋转：                  `transforms.RandomRotation`
- **对图像进行其他变换：**
  - :star:上 / 下采样（Resize）：       `transforms.Resize`
  - :star:将 PIL 图像转为 Tensor ：     `transforms.ToTensor`
  - 将 Tensor 转为 PIL 图像：            `transforms.ToPILImage`
  - :star:标准化：                                       `transforms.Normalize`
  - 填充：                                                  `transforms.Pad`
  - 修改亮度、对比度、饱和度：    `transforms.ColorJitter`
  - :star:将图像转化为灰度图：           `transforms.Grayscale`
  - 以概率 $p$ 将图像转化为灰度图：`transforms.RandomGrayscale(p=0.5)`
  - 线性变换：                                         `transforms.LinearTransformation`
  - 仿射变换：                                         `transforms.RandomAffine`
  - 自定义变换：                                    `transforms.Lambda`

- **对 Compose 的处理，使数据增强更灵活：**
  - 从许多 transform 操作中随便选一个操作：  `transforms.RandomChoice(transforms)`
  - 对于某一个 transform 操作以 $p$ 概率执行：  `transforms.RandomApply(transforms, p=0.5)`
  - 打乱 transform 操作的顺序：                              `transforms.RandomOrder`



###### 2.2.1 裁剪部分

1. 随机裁剪函数：`transforms.RandomCrop`，从图像中随机裁剪出一个符合`size`参数要求的图像

```py
torchvision.transforms.RandomCrop（
	size，	                 # 必须输入项，要么是 1 个 int 数，要么是 (h, w) 序列
    
    padding       = None，	 # 图像每个边框上的可选填充。默认值为None，即无填充。
    						  # 如果提供长度为4的序列，则它用于分别填充左，上，右，下边界。
        					  # 如果提供长度为2的序列，则分别用于填充左/右，上/下边界
    
    pad_if_needed = False，	 # 如果小于所需大小，它将填充图像以避免引发异常
    
    fill          = 0，	     # 恒定填充的填充值.如果长度为3的元组，则分别用于填充R，G，B通道
    
    padding_mode  ='constant' # 填充类型，这里提供了 4 种填充模式：
                              # 1.constant，常量（需要指定前面的参数'fill'）；
        					  # 2.edge，按照图片边缘的像素值来填充
            				  # 3.reflect，具有图像反射的垫（不重复边缘上的最后一个值）
                			  # 4.symmetric，具有图像反射的垫（重复边缘上的最后一个值）
）
```

2. 中心裁剪函数：`transforms.CenterCrop`，从图像中心裁剪出一个符合`size`参数要求的图像

```py
torchvision.transforms.CenterCrop(
    size	# 必须输入项，要么是 1 个 int 数，要么是 (h, w) 序列
) 
```

3. 随机长宽比裁剪函数：`transforms.RandomResizedCrop`，就是先将给定的图像进行随机放缩（比例范围由 `scale` 确定），再对图像的宽高比例进行随机放缩（宽高的比例范围由 `ratio` 确定）。在上面两个操作做完后，再从图像中中心裁剪出符合 `size` 参数要求的图像

```py
torchvision.transforms.RandomResizedCrop(
    size, 					# 必须输入项，要么是 1 个 int 数，要么是 (h, w) 序列
    scale=(0.08, 1.0), 		# 裁剪的原始尺寸的大小范围
    ratio=(0.75, 1.3333333333333333), # 裁剪的原始宽高比的宽高比范围
    interpolation=2			# 差值方法：默认 2（这个 2 实际上是 PIL.Image.BILINEAR）
)
```

4. 上下左右中心裁剪函数：`transforms.FiveCrop`，就是把一张图的上、下、左、右、中间这五个地方各自裁剪出一张符合 `size` 参数要求的图像（相当于 1 张图变成 5 张图）

```py
torchvision.transforms.FiveCrop(
    size	# 必须输入项，要么是 1 个 int 数，要么是 (h, w) 序列
)
```

> Note：如果对 1 张图像进行 `FiveCrop` 进行处理，会得到 5 张图，而且这些图会被整合到一个 5D 的 Tensor 上（`torch.Size(batchsize，ncrops，c，h，w)`），在后续的处理中还需要把这个 5 维的Tensor转成四维（通常可以用`tensor.view()`来实现），例如：

```py
bs, ncrops, c, h, w = input.size()
result = model(input.view(-1, c, h, w)) # 将 ncrops 和 batchsize 整合到一起
```

5. 上下左右中心裁剪后翻转函数：`transforms.TenCrop`，和前面的`transforms.FiveCrop`类似，就是相当于在`transforms.FiveCrop`得到的 5 张图的基础上对每张图做一下水平翻转或者垂直翻转，变成 10 张图

```py
torchvision.transforms.TenCrop(
    size, 					# 必须输入项，要么是 1 个 int 数，要么是 (h, w) 序列
    vertical_flip=False		# True 就用垂直翻转，False 就用水平翻转
)
```



###### 2.2.2 翻转和旋转部分

1. **【翻转】**以概率 $p$ 水平翻转函数：`transforms.RandomHorizontalFlip(p=0.5)`，以给定的概率 $p$ 对输入图像进行随机水平翻转【左右对调】

```py
torchvision.transforms.RandomHorizontalFlip(
    p=0.5	# 操作概率，默认值 0.5
)
```

2. **【翻转】**以概率 $p$ 垂直翻转函数：`transforms.RandomVerticalFlip(p=0.5)`，以给定的概率 $p$ 对输入图像进行随机垂直翻转【上下对调】

```python
torchvision.transforms.RandomVerticalFlip(
    p=0.5	# 操作概率，默认值 0.5
)
```

3. **【旋转】**随机旋转函数：`transforms.RandomRotation`，按照给定的角度范围对图像进行随机旋转（对于处理的图像而言，转是一定会转的，每张图转多少度是随机的，随机的范围取决于输入的参数`degrees`）

```python
torchvision.transforms.RandomRotation(
    degrees, 		# 随机旋转的度数范围，填一个int值x时就是±x°，填元组的话就按照(min,max)这样算
    
    resample=False, # 重采样时使用的过滤器类型，可选项包括：
    				# 1) PIL.Image.NEAREST【输入图像是灰度图或 8bit 彩色图会强制用这种过滤器】
    				# 2) PIL.Image.BILINEAR
    				# 3) PIL.Image.BICUBIC
    				# 默认为 False
    
    expand=False, 	# 如果为 False，旋转后的图像会保持原来的size，但原图中的部分内容被“转出去了”
    				# 如果为 True，旋转后要保证原图所有信息都在，这意味着旋转后的图像 size 会变大
    
    center=None,	# 旋转中心点，默认为中心
    				# 如果想手动指定也可以给定一个元组坐标 (x, y)
    
    fill=None		# 是否填充因为旋转多出来的“黑边”（默认不填充）
    				# 如果要填充的话需要在这里给定一个 RGB (取值范围 0-255) 的三元组：(R,G,B)
)
```



###### 2.2.3 对图像进行其他变换

1. 上 / 下采样函数： `transforms.Resize`，将输入的 PIL 图像的大小调整为给定大小

```py
torchvision.transforms.Resize(
    size, 				# 必须输入项，要么是 1 个 int 数，要么是 (h, w) 序列
    					# 在输入1个数时，原图的宽高中较小的会被设定为 size，而较大的则会按比例调整
    
    interpolation=2,	# 差值方法，默认的 2 其实就是 PIL.Image.BILINEAR
    					
    max_size=None, 		# 调整后的图像宽高中较大的边的最大值
    					#（就是如果设定的 size 比这里的 max_size 大，则再次调整图像，使得这个长边大小为 max_size）
    antialias=None		# 抗锯齿标志，通常不做改动
)
```

2. 将 PIL 图像转为 Tensor 函数： `transforms.ToTensor`，将取值范围为 0~255 的 PIL 图像转为取值范围为 0~1 的 Tensor。

   > Note： PIL 库打开的图像，原始的 shape 为$(H \times W \times C)$，而经过`transforms.ToTensor`函数处理后的 Tensor 的 shape 为 $(C \times H \times W)$

```py
torchvision.transforms.ToTensor()	# 通常不需要在里面定参数
```

3. 将 Tensor 反转回 PIL 图像函数： `transforms.ToPILImage`，将 Tensor 或者 np.ndarray 的数据转换为 PIL  类型图像

```py
torchvision.transforms.ToPILImage(
    mode=None	# 默认为 None，是单通道（mode=3 就是转换为 RGB 图，mode=4 就是转换为 RGBA 图）
)
```

4. Tensor 标准化函数：`transforms.Normalize`，用平均值和标准偏差归一化张量图像

```py
torchvision.transforms.Normalize(
    mean, 	# 每个通道的均值序列元组   （要处理的图像有多少通道，这里就要给多少个值）
    std		# 每个通道的标准偏差序列元组（要处理的图像有多少通道，这里就要给多少个值）
)
```

5. PIL 图像填充函数 `transforms.Pad`，使用给定的“pad”值在所有面上填充给定的 PIL 图像，就是把原图放在一个**新的画布**（大小由`padding`参数确定）里面，然后根据`fill`参数和`padding_mode`参数来决定如何来填充新画布中的黑边区域：

```py
torchvision.transforms.Pad(
    padding, 	# 每个边框上的填充
    			# 如果提供单个 int，则用于填充所有边框
    			# 如果提供长度为 2 的元组，则分别为 左/右 和 上/下 的填充
    			# 如果提供长度为 4 的元组，则分别为 左，上，右和下边框的填充
    
    fill=0, 	# 只有当下面的 padding_mode 参数取值为为'constant'时这个fill才有用
    			# 如果只给一个常量，那这个数就是用来填充像素的值，默认值为0
    			# 如果长度为 3 的元组，则分别用于填充 R，G，B 通道
    
    padding_mode='constant'	# 填充类型，这里提供了 4 种填充模式：
                            # 1.constant，常量（需要指定前面的参数'fill'）；
        					# 2.edge，按照图片边缘的像素值来填充
            				# 3.reflect，具有图像反射的垫（不重复边缘上的最后一个值）
                			# 4.symmetric，具有图像反射的垫（重复边缘上的最后一个值）
)
```

6. 修改 PIL 图像的亮度、对比度、饱和度的函数： `transforms.ColorJitter`，根据输入的参数范围来随机更改图像的亮度，对比度和饱和度

```py
torchvision.transforms.ColorJitter(
    brightness=0, 	# 亮度,[max(0,1-brightness)，1+brightness]或给定[min，max]来随机取值
    contrast=0, 	# 对比度,[max(0,1-contrast)，1+contrast]或给定[min，max]来随机取值
    saturation=0, 	# 饱和度,[max(0,1-saturation)，1+saturation]或给定[min，max]来随机取值
    hue=0			# 色相,[-hue，hue]或给定的[min，max]来随机取值
)
```

7. 将 RGB 图像转化为灰度图像函数：`transforms.ColorJitter`

```py
torchvision.transforms.Grayscale(
    num_output_channels=1	# 参数为 1 时，正常的灰度图，当为 3 时， 就制作 R=G=B 的通道图
)
```

8. 以概率 $p$ 将图像转化为灰度图函数：`transforms.RandomGrayscale(p=0.5)`，对于三通道的RGB图像，函数直接将图像转化为 R=G=B 的灰度图

```py
torchvision.transforms.RandomGrayscale(
    p=0.1	# 转化概率
)
```

9. 线性变换函数：`transforms.LinearTransformation`，使用方形变换矩阵和离线计算的`mean_vector` 变换张量图像。给定`transformation_matrix`和`mean_vector`，将使矩阵变平。从中拉伸并减去`mean_vector`，然后用变换矩阵计算点积，然后将张量重新整形为其原始形状。

   > Note： 白化转换：假设$X$是列向量零中心数据。然后torch.mm计算数据协方差矩阵[D x D]，对该矩阵执行SVD并将其作为transformation_matrix传递

```py
torchvision.transforms.LinearTransformation(
    transformation_matrix,	# 张量[D x D]，D = C x H x W.
    mean_vector				# 张量[D]，D = C x H x W.
) 
```

10. 仿射变换函数：`transforms.RandomAffine`，对图像保持中心不变的随机仿射变换

```py
torchvision.transforms.RandomAffine(
    degrees, 	# 要选择的度数范围，设定 0 可停用旋转
    			# 如果 degrees 是一个数字而不是像（min，max）这样的序列，则度数范围将是（-degrees，+degrees）
    
    translate=None, # translate（元组，可选） - 
    				# 水平和垂直平移的最大绝对分数元组
    				# 例如 translate =（a，b），然后在范围-img_width * a <dx <img_width * a中随机采样水平移位
    				# 并且在-img_height * b <dy <img_height * b范围内随机采样垂直移位。默认情况下不会翻译

    
    scale=None, 	# 缩放因子间隔，例如（a，b），然后从范围a<=scale<=b中随机采样缩放。默认情况下会保持原始比例
    
    shear=None, 	# 要选择的度数范围
    				# 如果degrees是一个数字而不是像（min，max）这样的序列，则度数范围将是（-degrees，+ degrees）
    
    resample=False, # # 重采样时使用的过滤器类型，可选项包括：
    				# 1) PIL.Image.NEAREST【输入图像是灰度图或 8bit 彩色图会强制用这种过滤器】
    				# 2) PIL.Image.BILINEAR
    				# 3) PIL.Image.BICUBIC
    				# 默认为 False
    
    fillcolor=0		# 输出图像中变换外部区域的可选填充颜色（也是要求RGB三个通道一个通道一个值）
) 
```

11. 自定义变换函数：`transforms.Lambda`

```py
torchvision.transforms.Lambda(
    lambda	# 必须指定的自定义 lambda 函数
)
```



###### 2.2.4 对 Compose 的处理，使数据增强更灵活
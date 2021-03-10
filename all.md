# 快速开始 {moduledoc0_0}
## 关于模块 {moduledoc0_1}
### 1. 创建与开发
模块是可复用的算法组件，可被引用在无限多个项目中。如果平台中还没有你所需要的公开模块，那么是时候“自己动手，丰衣足食”啦。点击`新建模块`开始创建。如果你想进行全流程开发，选择“空白项目”；如果想直接导入 GitHub 中的模块，选择`从 GitHub 引入`。

模块开发的详细操作见 <span target_doc='如何开发模块' style="color:#0366d6;cursor:pointer;">如何开发模块</span>。
### 2. 部署
模块开发完成之后你就可以在项目中引用自己的模块了。未部署的模块只对自己可见，如果想让其他用户也能发现并引用你的模块，你需要将它进行部署。

部署详细操作参考 <span target_doc='如何部署模块' style="color:#0366d6;cursor:pointer;">如何部署模块</span>。
### 3. 公开使用
已部署的模块默认状态为公开。所有用户都可在 [发现](/explore?&type=hot&classification=all) 查看并引用你的模块。
## 4. 管理与更新
模块开发者有管理与更新模块的权利。你可以在 [工作台](/workspace?tab=module) 的模块列表中找到相应的模块，进入 Notebook 对该作品进行更新并部署新的版本，新版本部署成功之后关注此模块的用户会收到版本迭代通知。


## 从 github 导入 {moduledoc0_2}
如果你是一个人工智能和机器学习的资深玩家，想必一定有不少模型托管在 github 上。现在，你可以非常方便的将他们迁移到 mo 平台上，通过部署与公开让你的项目“活起来”，让更多的人基于此制造出更多更好玩的 App 。
点击`新建模块`，选择 `从 GitHub 引入`，输入你的 git repo 的地址，你的项目将自动的下载到新创建的模块项目中。 随后，进入 notebook 进行调试，部署，即可。


# 如何开发模块 {moduledoc1_0}
## 导入数据集 {moduledoc1_1}

机器学习任务中，数据集是至关重要的组成元素之一，如果对于你目前开发的模块，你暂时没有合适的数据集，那么可以尝试搜索他人公开的数据集来使用。

点击左侧的 Dataset 栏，按关键字或者热门标签查询相关的数据集。在结果列表中选择合适的数据集，点击打开数据集的详情页面，进一步查看文件列表和作者的说明文档。

点击 `Import` 按钮，此数据集将以只读模式自动挂载到`/datasets/<imported_dataset_directory> `路径下。

如果需要修改，可在 `Notebook(.ipynb)` 中使用 `!cp -R ./datasets/<imported_dataset_directory>  ./<your_folder>` 指令将其复制到其他文件夹后再编辑。

对于引入的数据集中的 `.zip` 文件，可使用 `!unzip ./datasets/<imported_dataset_dir>/<XXX.zip> -d ./<your_folder>` 指令解压缩到其他文件夹后使用。


## 分析数据 {moduledoc1_2}

接下来，你可以使用 numpy、pandas及其他工具对数据进行进一步的分析与探索。


## 编写训练和预测方法 {moduledoc1_3}

### 1. 导入需要的python模块

<img src='http://imgbed.momodel.cn/5ccea307e3067c14d880bbb6.jpg' width=60% height=40%>

一般机器学习算法包括训练和预测两部分，经过训练过程会产生一个模型文件，在预测时调用训练好的模型文件进行预测。所以我们在程序中分为 `train` ， `predict` 和 `load_model` 三种方法，作为示例，kmeans聚类算法的主程序如下图所示。
<img src='http://imgbed.momodel.cn/5ccea315e3067c14d880bbc9.jpg' width=100% height=100%>


### 2. 训练过程

设定训练模型的保存路径，训练数据等参数，在 Notebook 中运行以下代码，训练产生算法模型文件。

<img src ='http://imgbed.momodel.cn/5ccea310e3067c14d880bbc4.jpg' width=100% height=100%>

你可以看到在当前目录下产生了一个xx.pkl模型文件

### 3. 使用 BTB 调参工具包进行自动调参

你可以使用 BTB 调参工具包来对你的模型进行自动的调参。
下面以一个随机森林模型和mnist数据集来说明如何使用此工具。

首先在工具列表中选择 自动调参工具， 点击后可预览如何使用调参工具的代码，点击确认，如何使用调参工具的代码将自动插入到你目前已激活的 notebook 中，mo 工具包将自动导入到你的项目中，你可在左侧的文件列表中找到 mo 文件夹。

![](http://mo-imgs.momodel.cn/hyperparameter.gif)

然后，你需要修改如何使用调参工具的代码。我们按以下步骤进行：


1.定义需要调参的参数的信息，tunables


定义随机森林模型需要调参的参数的信息
```python
from btb import HyperParameter, ParamTypes
tunables = [
    ('n_estimators', HyperParameter(ParamTypes.INT_EXP, [1, 1e2])),
    ('max_depth', HyperParameter(ParamTypes.INT, [1, 10]))
]
```
注意：
可选的参数类型为 INT, INT_EXP, INT_CAT, FLOAT, FLOAT_EXP, FLOAT_CAT, STRING, BOOL

* INT ,整型，   例如 [1, 10] 会从1到10的均匀分布中选择整型数值
* INT_EXP,  例如 [1, 1000] 会从1到1000的指数型分布（以10为底）中选择整型数值
* INT_CAT,  例如 [1, 10, 1000] 会从列表的三个值中选择一个
* FLOAT, 浮点型,    例如 [1.0, 10.0] 会从1.0到10.0的均匀分布中选择浮点型数值
* FLOAT_EXP, 例如 [1.0, 1000.0] 会从1.0到1000.0的指数型分布（以10为底）中选择浮点型数值
* FLOAT_CAT, 例如 [1.0, 10.0, 1000.0] 会从列表的三个值中选择一个
* STRING, 字符串，例如 ["a", "b"，"c"] 会从列表的三个值中选择一个
* BOOL,布尔型


2.准备训练和测试数据，X, y, X_test, y_test

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X, X_test, y, y_test = train_test_split(
    X,
    y,
    train_size=1000,
    test_size=300,
)
```

3.定义优化的评价函数 `score_fun`， 输入为 `predicted`, `y_test`
注意整个调参的目标就是最小化此函数值

```python
from sklearn.metrics import accuracy_score
score_fun = accuracy_score
```


4.定义需要优化的模型 model

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
            n_jobs=-1,
            verbose=False,
        )
```
 
注意，模型需实现 set_params 方法，来更新新的超参数


5.开始进行调参
```python
from mo.hyperparameter import HyperparameterTuner

tu = HyperparameterTuner(X, y, X_test, y_test, tunables, score_fun, model, trail_numer=3)
best_model, best_para = tu.start_tune()
```

注意，因为调参可能需要花费较长的时间，建议创建 job 来进行模型的调参，并在 job 日志中查看调参的打印输出，更多关于 job 的信息，可参考[这里](http://www.momodel.cn:8899/docs/#/zh-cn/%E8%BE%B9%E5%AD%A6%E8%BE%B9%E5%81%9A?id=_4-%E5%88%9B%E5%BB%BA-gpu-job-%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B)。

### 4. 预测过程

输入需要预测的数据，加载训练好的模型，调用 `predict` 方法，得到预测结果。
<img src='http://imgbed.momodel.cn/5ccea30ce3067c14d880bbbf.jpg' width=80% height=80%>


## 在 GPU/CPU 资源上训练模型 {moduledoc1_4}

### 1. 简介
在 `Mo` 平台上你可以通过两种方式训练你的模型：在 Notebook 中直接运行、通过建立 Job 在 GPU / CPU 后台运行。前者在网页上长时间运行很容易因为各种外部因素而中断，适合短时间小模型的调试训练。后者则通过后台建立Job任务运行，而且可以选择GPU加速，适合长时间大模型的训练。

下面我们采用卷积神经网络对手写数据集 ([MNIST](http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html))进行分类，指引大家学习如何在 CPU / GPU 资源上训练机器学习模型，本教程采用卷积神经网络进行分类，相较于全连接神经网络具有更好的特征提取能力，能够更好地保存图片的二维特征，在手写数据集上的准确率也会有很大提升。你可以前往训练营- [平台功能教程](http://www.momodel.cn:8899/classroom/class?id=5c5696cd1afd9458d456bf54&type=doc) 选择“在 GPU 或 CPU 资源上训练机器学习模型
”开始学习，按照指引完成相应的操作。

### 2. 在 Notebook 中调试训练
首先我们运行下面的 Cell 代码进行必要模块的导入, 以及参数的定义, 我们在这里将每个 epoch 训练的的 batch_size 定为 128
<img src ='http://imgbed.momodel.cn/5ccea313e3067c14d880bbc7.jpg' width=100% height=100%>
然后我们导入数据集以及做一些数据预处理
<img src ='http://imgbed.momodel.cn/5ccea30fe3067c14d880bbc3.jpg' width=100% height=100%>
然后使用Keras的Sequential定义两层卷积网络模型
<img src ='http://imgbed.momodel.cn/5ccea309e3067c14d880bbb9.jpg' width=100% height=100%>
接下来运行下面的 Cell 代码进行训练
<img src ='http://imgbed.momodel.cn/5ccea308e3067c14d880bbb7.jpg' width=100% height=100%>

*如果觉得训练时间太长, 可以直接点击 Notebook 顶部的<img src='http://imgbed.momodel.cn/5ccea309e3067c14d880bbb8.jpg' width='30px'>按钮停止程序的运行, 然后到下一小节, 把以上代码转换为 py 类型的文件，通过创建 Job 任务的方式训练模型。*

最后保存训练好的模型
<img src ='http://imgbed.momodel.cn/5ccea30de3067c14d880bbc0.jpg' width=100% height=100%>

*PS: 这里有个重要的地方, 我们需要将模型保存到 `results/` 文件夹下, 因为这个文件夹是 Job 与 Notebook 的共享文件夹, Job中的训练结果只有保存到 `results/` 下才能被 Notebook 读取到。*

### 3. 导出代码为 Python 文件
由于加入了深层卷积网络, 此次训练过程可能会比较长, 我们不推荐在 Notebook 中进行长时间训练, 最好的方法是通过创建一个 GPU Job 后台训练模型。 
Notebook 中的代码是在 *.ipynb 文件下的，为之后创建 Job 和部署做准备，点击 <img src='http://imgbed.momodel.cn/5ccea30ce3067c14d880bbbe.jpg' width=3% height=3%> 将其转为 `.py` 格式的标准 python 代码。然后整理你的代码，完成测试后，即可进行下一步的操作。  

*如果你是从模版中创建的项目，我们已经为你准备好了一份整理好的 `How_Train_Model.py` 文件, 你可以从左侧 'Files' 文件目录中双击查看, 并直接进行下一步。*  

### 4. 创建 GPU Job 训练模型
点击 Python 编辑器上方的 Create Job, 选择 `GPU 机器`创建 Job ，我们可以选择为 Job 输入一个容易辨识名字，当然也可以选择不输入，系统会默认生成。你也可以创建 `Notebook 控制台` 或 `CPU 机器` 形式的 Job ，这需要根据你训练的模型特点选择。

<img src='http://imgbed.momodel.cn/GPU.gif' width=80% height=50%>


### 5. 查看 Job 运行进程
<img src='http://imgbed.momodel.cn/prepare.png' width=30% height=30%>
<img src='http://imgbed.momodel.cn/running.png' width=30% height=30%>
<img src='http://imgbed.momodel.cn/complete.png' width=30% height=30%>


## 利用 TensorBoard 可视化评估模型 {moduledoc1_5}

### 1. 简介
我们集成了名为 TensorBoard 的可视化工具，来帮助你理解、调试和优化你的 TensorFlow 机器学习模型。
你可以使用TensorBoard来可视化你的机器学习模型的推理图（graph）结构，显示训练过程中准确率和损失函数的变化，绘制关于图形执行的量化指标等等。下面以手写数据集MNIST为例，来帮助你在Mo平台上更便捷的使用 TensorBoard 可视化工具。更多关于 TensorBoard 的信息请参考[官方文档](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) 和 [官方 Github 开源项目](https://github.com/tensorflow/tensorboard)。你可以前往训练营- [平台功能教程](http://www.momodel.cn:8899/classroom/class?id=5c5696cd1afd9458d456bf54&type=doc) 选择“利用 TensorBoard 可视化评估模型
”开始学习，按照指引完成相应的操作。

### 2. 撰写模型代码
导入手写数据集，定义神经网络图结构和训练过程，并在程序代码中添加 TensorBoard 总结指令，具体可以参考 TensorBoard [官方文档](https://www.tensorflow.org/guide/summaries_and_tensorboard?hl=zh-cn)。这里需要注意产生的 logs 文件要保存在 `./results/tb_results`文件夹。
<img src='http://imgbed.momodel.cn/5ccea314e3067c14d880bbc8.jpg' width=80% height=80%>

### 3. 利用 TensorBoard 查看模型结构和训练情况
点击左侧 Running Tab 栏，点击 TensorBoard 图标按钮<img src='http://imgbed.momodel.cn/5ccea30ee3067c14d880bbc1.jpg' width=10% height=10%>打开 TensorBoard 网页界面。

<img src='http://imgbed.momodel.cn/Tensorboard.png' width=60% height=60%>

查看你的模型 Graph 和训练情况。

<img src='http://imgbed.momodel.cn/Tensorboard.gif' width=90% height=90%>

## 把模型转换为 TensorFlow Lite 格式 {moduledoc1_6}

### 1. 简介
TensorFlow Lite 是 TensorFlow 针对移动和嵌入式设备的轻量级解决方案。它赋予了这些设备在终端本地运行机器学习模型的能力，从而不再需要向云端服务器发送数据。这样一来，不但节省了网络流量、减少了时间开销，而且还充分帮助用户保护自己的隐私和敏感信息。本部分将指引你把训练好的机器学习模型转换为手机或嵌入式设备可以使用的 TensorFlow Lite 格式。你可以在训练营- [平台功能教程](http://www.momodel.cn:8899/classroom/class?id=5c5696cd1afd9458d456bf54&type=doc) 选择“把模型转换为 TensorFlow Lite 格式”进行实际操作。

### 2. 准备训练好的模型
当转换为 TensorFlow Lite 格式时，需要将模型和权重都保存在同一个文件中。如果你在 Tensorflow 框架下构建并训练你的模型，你需要将模型保存为 SavedModel 格式。SavedModel 是一种独立于语言且可恢复的序列化格式，使较高级别的系统和工具可以创建、使用和转换 TensorFlow 模型。创建 SavedModel 的最简单方法是使用 ```tf.saved_model.simple_save``` 函数。更多关于 SavedModel 的信息请见[官方文档](https://www.tensorflow.org/programmers_guide/saved_model)。

如果你在 Tensorflow Keras 框架下构建并训练你的模型，你可以使用 ```tf.keras.models.save_model``` 函数将模型和权重都保存在同一个模型文件中，这里我们通过构建一个 keras 简单模型并且保存训练好的模型，得到 ```.h5``` 后缀的模型文件。
<img src='http://imgbed.momodel.cn/5ccea30be3067c14d880bbbd.jpg' width=100% height=80%>
然后通过以下代码保存模型，会在当前文件目录中生成 'keras_model.h5' 文件。  
<img src='http://imgbed.momodel.cn/5ccea312e3067c14d880bbc6.jpg' width=100% height=80%>

### 3. 将模型转换为 TensorFlow Lite 格式
选择刚刚保存的 `.h5` 后缀的模型文件，然后右键选择 ```Convert To TensorLite``` 就可以得到转换之后的 TensorFlow Lite 模型文件。

<img src='http://imgbed.momodel.cn/afjsajfsl.jpg' width=50% height=50%>

### 4. 下载转换后的模型并将其嵌入你的本地程序
模型转换完成后，在左侧的文件列表中便可以看到以转换后的以 ```.tflite``` 为后缀的文件。
右键此文件，并点击下载，即可将其下载到你的本地电脑中。然后你可以将其嵌入到你的 Android app、iOS app 或 Raspberry Pi中。更多信息请参阅[TensorFlow官方文档](https://www.tensorflow.org/lite/devguide)

# 如何部署模块 {moduledoc2_0} 
## 插入主函数 {moduledoc2_1}
当你在 notebook 中已经开发完模块，接下来，就可以开始进行部署了。
在 Notebook 中点击左侧栏中的 `Deploy` 按钮，进入部署页面。

<img src='https://imgbed.momodel.cn/2019071610242.png' width=40% height=40%>
选择第一步的 `插入`按钮插入主函数方法。

<img src='https://imgbed.momodel.cn/2019071610804.png' width=40% height=40%>

## 准备部署文件 {moduledoc2_2}

点击第二步中的 `重新开始` 按钮，开始准备部署文件。

<img src='https://imgbed.momodel.cn/1563211032304.jpg' width=40% height=40%>

选择 notebook 中你需要的代码块，点击下一步生成代码预览，然后点击下一步，定义模型的输入输出参数。对于一般的模块来说，其包含了 `train` 方法和 `predict` 方法。我们在 `input` 和 `output` 中分别定义 `train` 方法和 `predict` 方法的输入参数的名称、类型、范围以及默认值。然后点击下一步，预览生成的 yml 文档。然后，点击完成。如果你发现前面操作有误，可以点击上一步来进行修正。

当然，你可以直接对 `main.py` 和 `module_spec.yml` 文件进行编辑。在左侧 `文件` 栏的文件列表中，你可以看到 `main.py` 和 `module_spec.yml` 文件，双击文件即可打开。你也可以点击左侧的 `Deploy` 栏中第三步的 `main.py` 和 `module_spec.yml` 来快速打开该文件进行编辑。


## 测试你的模块 {moduledoc2_3}

接下来，对编写好的程序进行测试，点击第三步中的 `test.ipynb` ，或者打开当前项目文件夹下的 `test.ipynb` 文件。

<img src='https://imgbed.momodel.cn/1563212690427.jpg' width=40% height=40%>

根据你模型的输入输出参数，补全下面的代码，对模型进行测试。

<img src='http://imgbed.momodel.cn/5cc1a0c3e3067ce9b6abf787.gif' width=90% height=90%>

测试通过后，可点击第四步 `部署`。

<img src='https://imgbed.momodel.cn/1563213381080.jpg' width=40% height=40% >

## 文件选择 {moduledoc2_4}

在开发中，你可能创建了很多不必要的中间文件，它们不是程序执行的必备文件。所以，部署时，勾选 `main.py` 文件及其他运行所必须的文件即可。需要注意的是，当勾选一个文件夹时，即代表改文件夹下的所有子文件夹及子文件都被选中了。
 
 
## 版本选择 {moduledoc2_5}

在进行部署之前，你还需要为你的模块选择合适的版本。当第一次发布模块时，版本号可以为 0.1.0，也可以是 1.0.0。当你的模块做了向下兼容的功能性新增的时候， 可以选择增加次版本号，也就是第二位的数字。当项目在进行了局部修改或 bug 修正时，可以选择增加修订号，也就是第三位的数字。需要注意的是，版本号是只增不减的，所以请谨慎选择版本号，再进行部署。


## 类别与标签的选择 {moduledoc2_6}

部署完成之后模块状态即变为公开，将展示在 [发现](/explore?&type=hot&classification=all) 页面。为了让大家更方便的找到你的模块，你可以进入模块详情页面编辑相关信息，为你开发的模块选择一个合理的领域。目前可选的领域有四类：计算机视觉（CV）， 自然语言处理（NLP），数据处理和其他。

此外，你还可以为模块添加标签。如果没有合适的推荐标签，你也可以创建自定义标签。


## 试试部署你的第一个模块 {moduledoc2_7}
在 mo 平台构建一个全新的模块也很简单，首先，点击`新建模块`，选择`空白项目`，即可轻松创建新的模块项目。创建完成后， 将自动进入 notebook， 这时，即可进行开发了。
这里，我们以开发一个简单的数据处理功能（ min-max 转换）为例，介绍如何快速开发一个模块。
进入模块项目的 notebook 后，在 Untitled.ipynb 中，
开发功能
```
input_list = [1,2,3,4]

import numpy as np
input_list = np.array(input_list)
max_value = max(input_list)
min_value = min(input_list)
res = (input_list-min_value) / (max_value-min_value)
res.tolist()
```

开发完毕后，双击 main.py, 这个文件包含了模块部署的标准代码框架。可以看到，模块以 class 的形式存在，我们根据模块的实际需要，去编写对应的 `train`， `predict` 和  `load_model` 方法。因为我们的 `min-max` 缩放转换功能不需要用到机器学习模型，所以不需要开发 `train` 方法和 `load_model` 方法。我们在 `predict` 方法中，填入刚才我们开发的代码。注意，我们需要从 `conf` 中获取我们的输入参数。然后把处理结果以 `key value` 的形式返回。
```python
    def predict(self, conf={}):
        input_list = np.array(conf['input_list'])
        max_value = max(input_list)
        min_value = min(input_list)
        res = (input_list-min_value) / (max_value-min_value)
        res = res.tolist()
        return {'res': res}
```


完成后，接下来，编写 `module_spec.yml`文件。`yml` 文件定义了模块的输入输出参数。我们在 `predict` 部分填写输入参数的名称、类型和默认值等信息。

```yaml
    predict:
    input_list:
     <<: *default
     value_type: "[float]"
     default_value: "[1.0,2.0,3.0,4.0,5.0]"
```

我们在 `output` 部分填写我们的输出的名称和类型信息。
```yaml
    output:
      predict:
       res:
         value_type: "[float]"
```


接下来,可以对 `main.py` 进行测试了。在左侧的文件列表中双击打开 `test.ipynb` 。这是我们的测试文件。在这里，我们将模拟模块部署后被调用的情况。因为我们开发的 `min-max` 转换模块不涉及 ```train``` 方法，所以无需对测试 ```train``` 方法。我们使用如下代码对 ```predict``` 方法进行一个简单的测试。


```python
input_list = [1,2,3,4,5]
res = module.predict({"input_list": input_list})
print(res)
```

如果看到打印出如下结果，说明模块通过了测试。
```python
{'res': [0.0, 0.25, 0.5, 0.75, 1.0]}
```

测试通过后，我们就可以进行部署了。点击左侧状态栏的 `部署` ， 点击第四步，`部署`。
选择 `main.py` ，`module_spec.yml`，`OVERVIEW.md`等部署需要的文件， 选择发布的版本为 `0.0.1` ，选择项目领域为 `数据处理` ，然后点击部署。至此，我们就完成了一个简单的模块的开发和部署。


# 其他 {moduledoc3_0}
## 常见问题 {moduledoc3_1}
### 1. 项目、应用和模块有什么区别？

项目是用户在 [工作台](/workspace?tab=app) 创建的软件程序的总称，包含已部署和未部署两种状态。项目中可以插入数据集和模块。用户可设定项目为公开或私有，公开项目在 [发现](/explore?&type=hot&classification=all) 展示。

应用是已部署的项目，公开的应用在 [应用中心](/appcenter) 展示。

模块中可以插入数据集，但不可插入其他模块或项目。模块可以被引用到项目中。已部署模块默认为公开，展示在 [发现](/explore?&type=hot&classification=all)。

### 2. 项目是否可以被部署成模块？

项目具有定向性，当前项目只能被部署成应用。如果想在已有项目的基础上开发模块，可以进入项目详情页面，选择“复制成模块”，即可进行模块开发。

## 开发者交流 {moduledoc3_2}
Mo 的目标是“降低人工智能开发门槛”，我们希望构建一个友好而有活力的开发者社区。
欢迎你加入 Mo 的社交网络群，和志同道合的朋友们交流想法，互相答疑、分享资源与经验。工作人员也在群内，有问题和意见可以随时提出，我们会尽快回应。

**开发者用户交流QQ群**：495770120

**浙江大学Python兴趣群**：651025483
 <div>
    <h3>微信公众号: MomodelAI</h3>
    <img src='https://mo-imgs.momodel.cn/WeChatPublicAccount.png' height="220"/>
 </div>
 <br/>
<div>
    <h3>添加管理员微信加入AI俱乐部群聊</h3>
    <img src='https://mo-imgs.momodel.cn/wechat_office_contact.png' height="210">
</div>




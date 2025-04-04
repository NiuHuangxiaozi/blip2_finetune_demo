<!--
 * @Author: riverman nanjing.com
 * @Date: 2025-04-04 00:51:26
 * @LastEditors: riverman nanjing.com
 * @LastEditTime: 2025-04-04 10:37:16
 * @FilePath: /wsj/bliptime/blip2_finetune_demo/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# blip2 模型在m4_daily数据集上的微调



## 1. 项目结构
```
.
├── dataset   #数据集保存的文件夹
│   └── m4_daily_decomposition
│       ├── season
│       │   ├── data-00000-of-00001.arrow
│       │   ├── dataset_info.json
│       │   └── state.json
│       └── trend
│           ├── data-00000-of-00001.arrow
│           ├── dataset_info.json
│           └── state.json
├── demo_picture.png 
├── finetuned_model #微调后保存的模型参数
│   └── blip2_2_7b
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── README.md
├── inference.py #推理代码
├── main_parallelism.py #并行的运行微调代码
├── main.py #串行的运行微调代码
├── models #模型参数的保存位置
│   └── blip2
└── pytorch_image2text_blip2_loss_curve.png #微调画出的loss曲线
```

## 2. 如何运行代码。
### 2.1 创建conda环境
```
conda create --name timedis python==3.10
```

### 2.2 下载代码
```
git clone git@github.com:NiuHuangxiaozi/blip2_finetune_demo.git
```
### 2.3 下载数据集
访问南大云盘，然后下载zip文件到dataset路径下：
```
unzip m4_daily_decomposition.zip
```
### 2.4 下载blip2_2.7b模型（推荐使用下面的方式，依次输入以下的命令）
在命令行一次输入下面的指令
```
1. export HF_ENDPOINT=https://hf-mirror.com
2. huggingface-cli login 在你的huggingface账户上获得相应的access token，然后粘贴过来，一路回车然后登陆成功
3. 进入models文件夹 mkdir blip2
4. cd models/blip2/
5. 输入 huggingface-cli download --resume-download Salesforce/blip2-opt-2.7b --local-dir . --local-dir-use-symlinks False --resume-download，等待模型下载完毕

```
### 2.5 下载安装包
```
pip install -r requirements.txt 
```
### 2.6 运行微调程序
指定微调3轮，batchsize为16（进行调整占满整个GPU集群）
```
bash finetune,sh
```
### 2.4 微调的结果
在bliptime/finetuned_model/blip2_2_7b里可以看到peft保存的模型参数,还会输出一个finetune_log.txt文件


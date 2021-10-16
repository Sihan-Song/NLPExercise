## exerciseProject1

本项目实现了TextCNN用来进行文本分类。

#### data.py

定义了数据的预处理操作，利用torchtext来完成，从Field进行分词，到自定义Dataset，再到构架词汇表，最后使用Iterator来生成Batch。

##### TODO

如何使用预训练的词向量，例如Glove。

试着使用更多的数据集来进行分类。



#### models.py

定义了TextCNN模型。



#### train.py

模型训练的基本流程，以及模型的评估。



#### config.py

一些基本的参数设置



#### data/

使用的数据集



| pretrained_embedding | multi_channel | test_acc |      |
| -------------------- | ------------- | -------- | ---- |
| False                | False         |          |      |
| True                 | False         |          |      |
| True                 | True          |          |      |


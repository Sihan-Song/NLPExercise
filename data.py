import torchtext
from torchtext.data import Field,Dataset,Example,TabularDataset
from torchtext.vocab import Vectors
import pickle
import os
import numpy as np
import pandas as pd

def text_tokenize(x):
    return [w for w in x.split(" ") if len(w)>0]

def label_tokenize(x):#将一个标签转化为了一个shape为1的数组
    return [x.replace("__label__","")]


#Field对输入的文本进行一些预处理，比如对每一个example进行某些操作

#先使用torchtext.data.Dataset来加载语料库，会调用field来进行分词
#然后调用Field.build_vocab来构建词汇表
#然后之后通过DataLoader来进行batching操作。

#在FastTextData中做的事情：生成example和定义相应的field
#有一些自定义的Dataset类，例如TabularDataset可以处理json, csv, tsv等文件
class FastTextDataset(Dataset):
    def __init__(self,path,text_field,label_field,sep='\t'):
        fields=[('text',text_field),('label',label_field)]
        examples=[]
        with open(path,'r') as f:
            for line in f:
                s=line.strip().split(sep)
                assert len(s)==2
                text,label=s[0],s[1]
                #e = torchtext.data.Example()
                #setattr(e, "text", text_field.preprocess(text))
                #setattr(e, "label", label_field.preprocess(label))
                e=Example.fromlist([text,label],fields)#创建 torchtext.data.Example 的时候，会调用 field.preprocess 方法
                examples.append(e)
        super(FastTextDataset, self).__init__(examples,fields)
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

def fasttextdata_prepare(file_path,batch_size,shuffle=False,train=True):
    text_field = torchtext.data.Field(sequential=True, tokenize=text_tokenize, lower=True)
    label_field = torchtext.data.Field(sequential=False, tokenize=label_tokenize, lower=True)
    if train:#训练数据需要建立词汇表并保存
        dataset=FastTextDataset(file_path,text_field,label_field)
        #glove_vectors = Vectors(name='/Disk/D02/songsihan/NLP/pretrainedVectors/glove/glove.6B.300d.txt',
        #                        cache='/Disk/D02/songsihan/NLP/pretrainedVectors/glove/')
        text_field.build_vocab(dataset)#为输入建立词汇表
        label_field.build_vocab(dataset)
        #之后可以通过text_field或者label_field来访问vocab。
        if not os.path.exists('vocab/'):
            os.makedirs('vocab/')
        save_vocab(text_field.vocab, 'vocab/text.vocab')  # 这里保存词汇表是因为词汇表在test过程中会用到
        save_vocab(label_field.vocab, 'vocab/label.vocab')
        train_iterator=torchtext.data.BucketIterator(dataset,batch_size,
                                                     sort_key=lambda x:len(x.text),
                                                     shuffle=shuffle)#这里的这个text应该是和上面Dataset中Field的名字是相对应的。
        return train_iterator,text_field,label_field
    else:#说明现在是测试阶段，需要加载词汇表
        #应该使用训练阶段得到的词汇表以及向量表
        dataset=FastTextDataset(file_path,text_field,label_field)
        #text_field.build_vocab(dataset)
        #label_field.build_vocab(dataset)
        text_field.vocab=load_vocab('vocab/text.vocab')
        label_field.vocab=load_vocab('vocab/label.vocab')
        test_iterator = torchtext.data.BucketIterator(dataset, batch_size,
                                                       sort_key=lambda x: len(x.text),
                                                       shuffle=False)
        return test_iterator,text_field,label_field

def save_vocab(vocab,filename):
    with open(filename,'wb') as f:
        pickle.dump(vocab,f)
def load_vocab(filename):
    with open(filename, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def IMDB_textTokenize(x):
    x=x.strip('\"')
    return [w for w in x.split(" ") if len(w)>0]
def IMDB_labelTokenize(x):
    return x.strip()
#处理一下IMDB Dataset，并使用预训练的vector，看看结果
def IMDBData_split():
    df=pd.read_csv('data/IMDB Dataset.csv')
    print(df.shape)
    train_data=df.sample(frac=0.9,axis=0)
    test_data=df[~df.index.isin(train_data.index)]
    print(train_data.shape[0],test_data.shape[0],df.shape[0])
    assert train_data.shape[0]+test_data.shape[0]==df.shape[0]
    train_data.to_csv('data/IMDB_train_data.csv',index=False)
    test_data.to_csv('data/IMDB_test_data.csv',index=False)

def prepareIMDBData(file_path,batch_size,shuffle=False,train=True,pretrained_embedding=False):
    #先定义field
    text_field=torchtext.data.Field(sequential=True,tokenize=IMDB_textTokenize,lower=True)
    label_field=torchtext.data.Field(sequential=False,tokenize=IMDB_labelTokenize,lower=True)
    fields=[('text',text_field),('label',label_field)]
    #接下来得到dataset。
    dataset=TabularDataset(path=file_path,format='csv',fields=fields,skip_header=True)
    if train:
        if pretrained_embedding:#表示使用预训练的Glove
            glove_vectors = Vectors(name='/Disk/D02/songsihan/NLP/pretrainedVectors/glove/glove.6B.300d.txt',
                            cache='/Disk/D02/songsihan/NLP/pretrainedVectors/glove/')
            text_field.build_vocab(dataset,vectors=glove_vectors)
        else:
            text_field.build_vocab(dataset)

        label_field.build_vocab(dataset)
        save_vocab(text_field.vocab, 'vocab/IMDB_text.vocab')  # 这里保存词汇表是因为词汇表在test过程中会用到
        save_vocab(label_field.vocab, 'vocab/IMDB_label.vocab')
        train_iterator = torchtext.data.BucketIterator(dataset, batch_size,
                                                       sort_key=lambda x: len(x.text),
                                                       shuffle=shuffle)  # 这里的这个text应该是和上面Dataset中Field的名字是相对应的。
        return train_iterator, text_field, label_field
    else:
        text_field.vocab=load_vocab('vocab/IMDB_text.vocab')
        label_field.vocab = load_vocab('vocab/IMDB_label.vocab')
        test_iterator = torchtext.data.BucketIterator(dataset, batch_size,
                                                      sort_key=lambda x: len(x.text),
                                                      shuffle=False)
        return test_iterator,text_field,label_field
#将数据集划分为训练集和验证集
IMDBData_split()
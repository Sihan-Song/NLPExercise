import torch
import torch.nn  as nn
import torch.nn.functional as F


class TextCNNSingleDim(nn.Module):
    #num_classes,drop_out,embedding_pretrained,
    def __init__(self,config,weight_matrix):
        super(TextCNN, self).__init__()
        self.config=config
        self.embedding=nn.Embedding(config.vocab_size,config.embedding_size,padding_idx=config.vocab_size-1)
        if config.pretrained_embedding:#如果是预训练的话，那么就不更新参数了
            self.embedding=nn.Embedding.from_pretrained(weight_matrix,freeze=True)
        if config.multichannel:#如果是multichannel的话这个channel需要更新参数
            self.embedding_multi=nn.Embedding.from_pretrained(weight_matrix,freeze=False)

        #形状（batch_size、channels、height、width）用于nn.Conv2d输入，需要一个四维的张量
        conv_types=[2,3,4]
        self.convs=nn.ModuleList(nn.Conv2d(1,1,(size,config.embedding_size),stride=1,padding=0) for size in conv_types)
        #这里是定义了6个卷积核

        #maxpool1d的输入形状应该是3维的
        #然后接下来是max_polling层
        #由于句子长度不一致，所以没办法使用torch.nn.MaxPool2d()函数
        #直接使用max函数好了

        #接下来是全连接层
        self.fc=nn.Linear(len(conv_types),config.num_classes)
        self.dropout=nn.Dropout(config.drop_out)
    def forward(self,x):#输入的x是一个句子例如：[232,19,734,12,88...]，里面的每一个数字代表词汇表中的索引
        x1=self.embedding(x)#x1=[batch_size,sentence_length,embedding_size]
        x2=None
        if self.config.multichannel:
            x2=self.embedding_multi(x)
        if x2 is not None:
            x=[(F.relu(conv(x1)) + F.relu(conv(x2))).squeeze(2) for conv in self.convs]#x=[convs,batch_size,sentence_length-kernel_size]
        else:
            x1=x1.unsqueeze(1)#(batch_size,1,sentence_length,embedding_size)
            #经过conv函数后变成：(batch_size,1,sentence_length-kernel_size,1)
            #经过squeeze后变成：(batch_size,1,sentence_length-kernel_size)
            x=[F.relu(conv(x1)).squeeze(3) for conv in self.convs]
        #接下来取最大值
        #x=torch.unsqueeze(x,dim=3)#将最后一维压缩，现在的x=[convs,batch_size,1,sentence_length-kernel_size]
        #x是一个list，list的每一个元素是一个tensor，该tensor的shape为（batch_size, 1, sentence_length-kernel_size)
        x=[F.max_pool1d(item,item.shape[2]).squeeze(1) for item in x]#里面的每个item处理之后变成(batch_size,1,1)
        #squeeze之后变成(batch_size,1)
        #x.shape=(convs,batch_size,1)
        #将第一维和第二维换一下顺序，变成(batch_size,convs)
        res=None
        for item in x:#每个item是(batch_size,1)
            if res is None:
                res=item
            else:
                res=torch.cat((res,item),dim=1)
        x=self.fc(self.dropout(res))#(batch_size,num_classes)
        return x

#这个里面每种卷积核的channel可以是大于1的
class TextCNN(nn.Module):
    #num_classes,drop_out,embedding_pretrained,
    def __init__(self,config):
        super(TextCNN, self).__init__()
        self.config=config
        self.embedding=nn.Embedding(config.vocab_size,config.embedding_size)
        input_channel=1
        if config.pretrained_embedding:#如果是预训练的话，那么就不更新参数了
            print('hhh')
            self.embedding=nn.Embedding.from_pretrained(config.weight_matrix,freeze=True)
        if config.multichannel:#如果是multichannel的话这个channel需要更新参数
            print('www')
            self.embedding_multi=nn.Embedding(config.vocab_size,config.embedding_size).from_pretrained(config.weight_matrix,freeze=False)
            input_channel=2

        #形状（batch_size、channels、height、width）用于nn.Conv2d输入，需要一个四维的张量
        conv_types=config.conv_types
        self.convs=nn.ModuleList(nn.Conv2d(input_channel,config.channel_dim,(size,config.embedding_size),stride=1,padding=0) for size in conv_types)
        #maxpool1d的输入形状应该是3维的

        #接下来是全连接层
        self.fc=nn.Linear(len(conv_types)*config.channel_dim,config.num_classes)
        self.dropout=nn.Dropout(config.drop_out)
    def forward(self,x):#输入的x是一个句子例如：[232,19,734,12,88...]，里面的每一个数字代表词汇表中的索引
        if self.config.multichannel:
            x1=self.embedding(x)#(batch_size,sentence_length,embedding_size)
            x2=self.embedding_multi(x)#(batch_size,sentence_length,embedding_size)
            x=torch.stack([x1,x2],dim=1)#沿着一个新维度对输入的tensor进行连接，比如将多个二维张量凑成一个3维张量
            #(batch_size,2,sentence_length,embedding_size)
        else:
            x=self.embedding(x).unsqueeze(1)#(batch_size,1,sentence_length,embedding_size)

        #经过conv函数后变成：(batch_size,channel_dim,sentence_length-kernel_size,1)
        #经过squeeze后变成：(batch_size,channel_dim,sentence_length-kernel_size)
        x=[F.relu(conv(x)).squeeze(3) for conv in self.convs]

        #x=torch.unsqueeze(x,dim=3)#将最后一维压缩，现在的x=[convs,batch_size,channel_dim,sentence_length-kernel_size]
        #x是一个list，list的每一个元素是一个tensor，该tensor的shape为（batch_size, channel_dim, sentence_length-kernel_size)
        x=[F.max_pool1d(item,item.shape[2]).squeeze(2) for item in x]#里面的每个item处理之后变成(batch_size,channel_dim,1)
        #squeeze之后变成(batch_size,channel_dim)
        #x.shape=(convs,batch_size,channel_dim)#如何拼接成(batch_size,channel_dim*convs)
        res=None
        for item in x:#每个item是(batch_size,channel_dim)
            if res is None:
                res=item
            else:
                res=torch.cat((res,item),dim=1)
        #得到的x为(batch_size,channel_dim*convs)
        x=self.fc(self.dropout(res))#(batch_size,num_classes)
        return x
import data
from models.TextCNN import TextCNN
import torch
import torch.nn.functional as F

import argparse
from config import Config
import os
parser=argparse.ArgumentParser(description='train textCNN')


#接下来定义训练函数
def train(train_iter,model,conf):
    optimizer=torch.optim.Adam(model.parameters(),lr=conf.learning_rate)
    model.train()
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    if conf.cuda:
        model=torch.nn.DataParallel(model).cuda()

    best_acc=0
    for epoch in range(conf.epochs):
        total_loss=0.0
        total_correct=0
        total_count=0
        batch_cnt=0
        for batch in train_iter:
            data,target=batch.text,batch.label#(sentence_length,batch_size),(sentence_length)

            data=data.t()#转置为(batch_size,sentence_length)
            target=torch.sub(target,1)#让class从0开始

            if conf.cuda:
                data=data.cuda()
                target=target.cuda()

            optimizer.zero_grad()
            logit=model(data)#得到的logit为：(batch_size,num_classes)
            loss=F.cross_entropy(logit,target)
            loss.backward()
            optimizer.step()

            pred=(torch.max(logit,1))[1].view(target.size())#表示预测出的标签
            correct=(pred.data==target.data).sum()#得到的是一个只有一个元素的tensor
            total_loss+=loss.item()
            total_correct+=correct.item()
            total_count+=data.shape[0]
            assert data.shape[0]==batch.batch_size
            batch_cnt+=1

        total_loss/=total_count
        acc=total_correct/total_count
        test_loss,test_acc=eval(test_iter,model,conf)
        if test_acc>best_acc:#保存最好的一个模型
            best_acc=test_acc
            torch.save(model.state_dict(),'checkpoints/best.pth')
        print('Training epoch [%d/%d] - training loss: %.6f  '
              'training acc: %.4f  test loss: %.4f  test acc: %.4f'%(epoch, conf.epochs, total_loss, acc,test_loss,test_acc))

def eval(data_iter,model,conf):
        model.eval()
        test_loss=0.0
        test_correct=0
        test_count=0
        for batch in data_iter:
            data,target=batch.text,batch.label
            data=data.t()
            target=torch.sub(target,1)

            if conf.cuda:
                data=data.cuda()
                target=target.cuda()
            logit=model(data)
            test_loss+=F.cross_entropy(logit,target)
            pred=(torch.max(logit,1))[1].view(target.size())
            test_correct+=(pred.data==target.data).sum().item()
            test_count+=batch.batch_size
        test_loss/=test_count
        acc=test_correct/test_count
        return test_loss,acc

if __name__=='__main__':
    conf=Config()
    #进行数据的预处理
    #train_iter,text_field,label_field=data.fasttextdata_prepare(conf.train_file,conf.batch_size,shuffle=True,train=True)
    #test_iter,_,_=data.fasttextdata_prepare(conf.test_file,conf.batch_size,shuffle=True,train=False)
    train_iter, text_field, label_field = data.prepareIMDBData(conf.train_file, conf.batch_size, shuffle=True,
                                                                    train=True,pretrained_embedding=conf.pretrained_embedding)
    test_iter, _, _ = data.prepareIMDBData(conf.test_file, conf.batch_size, shuffle=True, train=False)

    if conf.pretrained_embedding:
        conf.weight_matrix=text_field.vocab.vectors
    #更新一下词汇表的大小和类别的大小
    conf.vocab_size=len(text_field.vocab)
    conf.num_classes=len(label_field.vocab)-1 #这里减1是因为有一个unk token
    textcnn=TextCNN(conf)
    train(train_iter,textcnn,conf)

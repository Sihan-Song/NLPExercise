import torch
class Config(object):
    cuda=True
    epochs=300
    batch_size=64
    shuffle=True

    learning_rate=0.001
    learning_momentum=0.9
    weight_decay=0.0001

    dropout_dim=128
    train_file='data/IMDB_train_data.csv'
    test_file='data/IMDB_test_data.csv'

    model_name = 'TextCNN'
    num_classes = 10
    drop_out = 0.5
    pretrained_embedding = True
    multichannel = False
    weight_matrix=None
    embedding_size = 300#表示词向量的维度
    vocab_size = 0#表示词汇表的大小
    conv_types=[3,4,5]
    channel_dim=100
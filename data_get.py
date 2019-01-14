import jieba
from gensim.models.word2vec import Word2Vec
import numpy as np
from keras.utils import to_categorical

def data_get():
    data = open("data/斗罗大陆.txt",encoding='utf-8').read()
    print(len(data))
    data = data.replace('\n','')
    data = data.replace('-','')
    data = data.replace(' ','')
    data_fenci = jieba.cut(data)
    list_words = []
    for i in data_fenci:
        list_words.append(i)
    all_words = [word for word in list_words]
    print("Article's lenth is {}".format(len(all_words)))

    return all_words

def train_test_split(all_words,word2vec_model):
    """构建训练集和测试集"""
    # 处理原始数据，让其变成一个很长的x，然后预测下一个单词
    # (1)将语料库中的listoflist的每一个list全部打开，将所有的词全部取出来，变成list，里面全是词
    raw_input = all_words[0:int((len(all_words) / 1000))]
    print("Input data has {} words!".format(len(raw_input)))
    print("Start split training data and test data!")
    # (2)将在word2vec字典里面存在的词都放进去
    text_stream = []
    for raw_word in raw_input:
        if raw_word in word2vec_model:
            text_stream.append(raw_word)
    print("Total word lenth is {}".format(len(text_stream)))
    # （3）构造训练集:这里将整个文章从头开始进行切，seq_lenth 就是 切的长度，
    # 然后将文章切成10，10，10....，这样前面作为训练，后面作为预测
    text_stream = text_stream
    seq_lenth = 50
    X = []
    y = []
    for i in range(0, len(text_stream) - seq_lenth):
        train_seq = text_stream[i:i + seq_lenth]
        predic_seq = text_stream[i + seq_lenth]
        X.append(np.array(word2vec_model[train_seq]))
        y.append(np.array(word2vec_model[predic_seq]))
    # 最后将输入转成lstm的输入数组形式 ：[样本数，时间窗口，特征]
    X = np.array(X)
    y = np.array(y)
    return X,y


if __name__ == '__main__':
    data_get()



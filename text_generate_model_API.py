import numpy as np
import jieba
import os
import keras
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM,Activation
from gensim.models.word2vec import Word2Vec
from data_get import data_get,train_test_split
from keras .callbacks import EarlyStopping

class LSTM_model():
    def __init__(self):
        self.seq_lenth = 50
        self.epoches = 5000
        self.batch_size = 100
        self.min_count = 5
        self.vector_word_size = 128
        self.word2vec_model_path = "model/word2vec_model.model"
        self.lstm_model_path = "model/text_generate_model.h5"
        self.path = "data/斗罗大陆.txt"
        self.all_words = self.data_get()
        if os.path.exists(self.word2vec_model_path):
            self.word2vec_model = Word2Vec.load(self.word2vec_model_path)
        else:
            self.word2vec_model = self.word2vec_generate()
        self.X, self.y = self.data_train_test_split()
        self.model_build()


    def word2vec_generate(self):
        self.word2vec_model = Word2Vec(all_words, size=self.vector_word_size, window=5, min_count=self.min_count)
        self.word2vec_model.save(self.word2vec_model_path)
        return self.word2vec_model

    def data_get(self):
        data = open(self.path, encoding='utf-8').read()
        print(len(data))
        data = data.replace('\n', '')
        data = data.replace('-', '')
        data = data.replace(' ', '')
        data_fenci = jieba.cut(data)
        list_words = []
        for i in data_fenci:
            list_words.append(i)
        self.all_words = [word for word in list_words]
        print("Article's lenth is {}".format(len(self.all_words)))
        return self.all_words

    def data_train_test_split(self):
        """构建训练集和测试集"""
        # 处理原始数据，让其变成一个很长的x，然后预测下一个单词
        # (1)将语料库中的listoflist的每一个list全部打开，将所有的词全部取出来，变成list，里面全是词
        raw_input = self.all_words[0:int((len(self.all_words) / 1000))]
        print("Input data has {} words!".format(len(raw_input)))
        print("Start split training data and test data!")
        # (2)将在word2vec字典里面存在的词都放进去
        text_stream = []
        for raw_word in raw_input:
            if raw_word in self.word2vec_model:
                text_stream.append(raw_word)
        print("Total word lenth is {}".format(len(text_stream)))
        # （3）构造训练集:这里将整个文章从头开始进行切，seq_lenth 就是 切的长度，
        # 然后将文章切成10，10，10....，这样前面作为训练，后面作为预测
        text_stream = text_stream
        self.X = []
        self.y = []
        for i in range(0, len(text_stream) - self.seq_lenth):
            train_seq = text_stream[i:i + self.seq_lenth]
            predic_seq = text_stream[i + self.seq_lenth]
            self.X.append(np.array(self.word2vec_model[train_seq]))
            self.y.append(np.array(self.word2vec_model[predic_seq]))
        # 最后将输入转成lstm的输入数组形式 ：[样本数，时间窗口，特征]
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        return self.X, self.y

    def model_build(self):
        """建立模型"""
        print("Building Model......")
        self.model = Sequential()
        self.model.add(LSTM(units=256, input_shape=(self.seq_lenth, self.vector_word_size), dropout_W=0.2, dropout_U=0.2))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.vector_word_size, activation="sigmoid"))
        optimizer = keras.optimizers.RMSprop(lr=0.01)
        self.model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        self.model.fit(self.X, self.y, batch_size=self.batch_size, epochs=self.epoches, callbacks=[earlystop])
        print(self.model.summary())
        self.model.save(self.lstm_model_path)

class text_predict_articles():
    def __init__(self):
        self.word2vec_model_path = "model/word2vec_model.model"
        self.lstm_model_path = "model/text_generate_model.h5"
        self.word2vec_model = Word2Vec.load(self.word2vec_model_path)
        self.model = load_model(self.lstm_model_path)
        self.seq_lenth = 50
        self.rounds = 50
        self.vector_word_size = 128


    def genrate_article(self, input_sentences):
        # 这里将所有的大写转成小写
        input_sentences_2 = input_sentences
        next_data = ""
        for ii in range(self.rounds):
            # 将所有的输入转成模型输入的形式(-1,seq_lenth,128)
            input_words = jieba.cut(input_sentences_2)
            input_words = [word for word in input_words]
            input_data = []
            input_data_real_word = []
            lenth = len(input_words) - self.seq_lenth
            for word in input_words[lenth:]:
                if word in self.word2vec_model:
                    single_input = self.word2vec_model[word]
                    input_data.append(single_input)
                    input_data_real_word.append(word)
                else:
                    input_data.append([0 for _ in range(self.vector_word_size)])
                    input_data_real_word.append(word)
            input_data = np.array(input_data)
            input_data = input_data.reshape(-1, self.seq_lenth, self.vector_word_size)
            predic_word = self.model.predict(input_data)

            top_prob_word = self.word2vec_model.most_similar(positive=predic_word, topn=1)
            input_sentences_2 += "" + top_prob_word[0][0]
            next_data += top_prob_word[0][0]
        print(input_sentences_2)
        return next_data

if __name__ == '__main__':
    lstm_model = LSTM_model()
    # lstm_model.data_get("data/斗罗大陆.txt")
    text_articles = text_predict_articles()
    input_sentences = "二十九年了，自从二十九年前他被外门长老唐蓝太爷在襁褓时就捡回唐门时开始，唐门就是他的家，而唐门的暗器就是他的一切。突然，唐三脸色骤然一变，但很快又释然了' \
       '十七道身影，十七道白色的身影，宛如星丸跳跃一般从山腰处朝山顶方向而来，这十七道身影的主人，年纪最小的也超过了五旬，一个个神色凝重，他们身穿的白袍代表的是内门，而" + "start"
    generated_articles = text_articles.genrate_article(input_sentences)
    print(generated_articles)
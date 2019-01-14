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



def word2vec_generate(all_words,min_count = 5,size = 128):
    word2vec_model = Word2Vec(all_words, size=size, window=5, min_count=min_count)
    word2vec_model.save("model/word2vec_model.model")
    return word2vec_model

def word2vec_load(model_path = "model/word2vec_model.model"):
    return Word2Vec.load(model_path)

def genrate_article(model,word2vec_model,input_sentences,seq_lenth = 50 ,rounds = 50):
    # 这里将所有的大写转成小写
    input_sentences_2 = input_sentences.lower()
    next_data = []
    for ii in range(rounds):
        # 将所有的输入转成模型输入的形式(-1,seq_lenth,128)
        input_words = jieba.cut(input_sentences_2)
        input_words = [word for word in input_words]
        input_data = []
        input_data_real_word = []
        lenth = len(input_words) - seq_lenth
        for word in input_words[lenth:]:
            if word in word2vec_model:
                single_input = word2vec_model[word]
                input_data.append(single_input)
                input_data_real_word.append(word)
            else:
                input_data.append([0 for _ in range(128)])
                input_data_real_word.append(word)
        input_data = np.array(input_data)
        input_data = input_data.reshape(-1,seq_lenth,128)
        predic_word = model.predict(input_data)

        top_prob_word = word2vec_model.most_similar(positive=predic_word,topn=1)
        input_sentences_2 += "" + top_prob_word[0][0]
        next_data.append(top_prob_word[0][0])
    print(input_sentences_2)

if __name__ == '__main__':
    """数据处理"""
    all_words = data_get()
    if os.path.exists("model/word2vec_model.model"):
        word2vec_model = Word2Vec.load("model/word2vec_model.model")
    else:
        word2vec_model = word2vec_generate(all_words)
    X, y = train_test_split(all_words,word2vec_model)

    # 常量值
    seq_lenth = 50
    epoches = 5000
    batch_size = 100

    """建立模型"""
    print("Building Model......")
    model = Sequential()
    model.add(LSTM(units=256, input_shape=(seq_lenth, 128), dropout_W=0.2, dropout_U=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="sigmoid"))
    optimizer = keras.optimizers.RMSprop(lr=0.01)
    model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    model.fit(X, y, batch_size=32, epochs = epoches,callbacks=[earlystop])
    print(model.summary())
    model.save("model/text_generate_model.h5")

    """预测结果"""
    init ="二十九年了，自从二十九年前他被外门长老唐蓝太爷在襁褓时就捡回唐门时开始，唐门就是他的家，而唐门的暗器就是他的一切。突然，唐三脸色骤然一变，但很快又释然了' \
       '十七道身影，十七道白色的身影，宛如星丸跳跃一般从山腰处朝山顶方向而来，这十七道身影的主人，年纪最小的也超过了五旬，一个个神色凝重，他们身穿的白袍代表的是内门，而"
    genrate_article(model, word2vec_model, init, seq_lenth)




from keras.models import load_model
from lstm_generate_model import word2vec_load,genrate_article

if __name__ == '__main__':
    """预测结果"""
    init = "长老们都没有说话，他们此时还没能从佛怒唐莲的出现中清醒过来。" \
           "两百年，整整两百年了，佛怒唐莲竟然在一个外门弟子手中出现，这意味着什么？这霸绝天下，连唐门自己人也不可能抵挡的绝世暗器代表的绝对是唐门另一个巅峰的来临"
    model = load_model("model/text_generate_model.h5")
    word2vec_model = word2vec_load("model/word2vec_model.model")
    init = init + "start"
    genrate_article(model, word2vec_model, init)
import os
import pymysql
from settings import *
from collections import Counter
import pickle
from tensorflow import keras
import numpy
import gensim


def get_synonyms(file_path):
    """
    返回同义词字典
    :return: dict[str, str]
    """
    dic = {}
    path = os.path.join(file_path, './synonyms.txt')
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() != '':
                words = line.strip().split(' ')
                target = words[0].strip()
                for i in range(1, len(words)):
                    dic[words[i].strip()] = target
    return dic


def get_stop_words(file_path) -> set:
    """
    返回停用词表
    :return:
    """
    stop_words_dir = os.path.join(file_path, './stop_words.txt')
    with open(stop_words_dir, 'r', encoding='utf-8') as f:
        stop_words: set = {word.strip() for word in f if word.strip() and word.strip() != ''}
    return stop_words


def getWordEmbedding(words, file_path='../word2vec/'):
    """
    按照words中的词，取出预训练好的word2vec中的词向量
    :param words: 训练集中文档频率大于5的词
    :param file_path: 包含训练好的词向量模型的路径
    :return:
    """
    path = os.path.join(file_path, './word2Vec.bin')
    wordVec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    vocab = []
    wordEmbedding = []

    # 添加 "pad" 和 "UNK", 以及初始化对应的词向量
    vocab.append("<pad>")
    vocab.append("<UNK>")
    wordEmbedding.append(numpy.zeros(embedding_dim))
    wordEmbedding.append(numpy.random.randn(embedding_dim))

    for word in words:
        try:
            vector = wordVec.wv[word]
            vocab.append(word)
            wordEmbedding.append(vector)
        except:
            print(word + "不存在于词向量中")

    return vocab, numpy.array(wordEmbedding)


def build_vocab(vocab_dir='./vocabulary.txt', threshold=8):
    """
    根据2009年初到2017年10月中的文本（分词后），选出文档频率大于threshold的词，构建词典，并保存到vocab_dir中
    :param vocab_dir:
    :param threshold:
    :return:
    """
    # 连接数据库
    db = pymysql.connect(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
                         port=MYSQL_PORT, charset='utf8')
    cursor = db.cursor()

    # 构建查询语句，并获取结果
    results = []
    for blogger_name in blogger_names:
        sql = "select * from processed_blogs_all_words " \
              "where blogger_name = \'%s\' and created_date between '2009-01-01 00:00:00' and '2017-10-16 00:00:00'" \
              % blogger_name
        try:
            cursor.execute(sql)
            processed_blogs_all_words = cursor.fetchall()
        except:
            print('Error when fetching blogs')
            return
        results.extend(processed_blogs_all_words)

    dic = {}  # 记录所有词的文档频率
    for processed_blog in results:
        temp_set = set()  # 记录当前文章有哪些词
        words = processed_blog[2].split()
        for word in words:
            temp_set.add(word)
        for word in temp_set:
            if word in dic.keys():
                dic[word] += 1
            else:
                dic[word] = 1

    # 排序并选出文档频率大于threshold的词
    sorted_dic = sorted(dic.items(), key=lambda d: d[1])
    sorted_dic.reverse()

    words = [word for word, value in sorted_dic if value > threshold]
    vocab, wordEmbedding = getWordEmbedding(words)
    with open('./wordEmbedding.pk', 'wb') as f:
        pickle.dump(wordEmbedding, f)

    # 写入到词汇表
    with open(vocab_dir, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write('%s\n' % word)


def read_vocab(file_path):
    """
    读取词汇表，并返回包含所有词汇的list，以及一个包含 词汇：编码 的字典
    :param file_path: 词典所在的文件夹路径
    :return:
    """
    path = os.path.join(file_path, './vocabulary.txt')
    with open(path, 'r', encoding='utf-8') as f:
        words = [x.strip().split()[0] for x in f.readlines() if x.strip() and x.strip() != '']

    word_to_id = dict(zip(words, range(len(words))))

    path = os.path.join(file_path, './wordEmbedding.pk')
    with open(path, 'rb') as f:
        wordEmbedding = pickle.load(f)

    return words, word_to_id, wordEmbedding


def get_date_returns(file_path: str) -> dict:
    path = os.path.join(file_path, './date.pk')
    with open(path, 'rb') as f:
        date = pickle.load(f)

    path = os.path.join(file_path, './next_three_day_returns.pk')
    with open(path, 'rb') as f:
        next_three_day_returns = pickle.load(f)

    date_returns = {k: v for k, v in zip(date, next_three_day_returns)}
    return date_returns


def process_blogs(blogs):
    """
    早评、午评短于200，且对应前一日的收评、当天收评短于300，则合并到对应位置，否则去掉

    删除长度小于250的早评、晚评
    :return:
    """
    # 将blogs中的每个元素的type从tuple变为list
    for i, blog in enumerate(blogs):
        blogs[i] = list(blog)

    index_to_be_removed = []
    for i, blog in enumerate(blogs):
        created_date: datetime.datetime = blog[3]
        date = created_date.date()
        time = created_date.time()
        # 处理早评
        if time.__lt__(datetime.time(9, 25, 0, 0)):
            # 长度小于200
            if len(blog[2].split()) <= 200:
                for index in range(i + 1, i + 4):
                    # 寻找符合条件的上一交易日，并将早评添加到其后面
                    if index < blogs.__len__() \
                            and date.__sub__(blogs[index][3].date()).days <= 3 and len(blogs[index][2].split()) <= 300:
                        blogs[index][2] = ' '.join(blogs[index][2].split() + blog[2].split()[1:])
                        break
            if len(blog[2].split()) <= 250:
                index_to_be_removed.append(i)
        elif time.__lt__(datetime.time(14, 30, 0, 0)):  # 处理午评
            # 长度小于200
            if len(blog[2].split()) <= 200:
                for index in range(i - 1, i - 3):
                    # 查看当天是否有收评，若有，检查长度是否小于300
                    if date.__eq__(blogs[index][3].date()) and len(blogs[index][2].split()) <= 300:
                        blogs[index][2] = ' '.join(blogs[index][2].split() + blog[2].split()[1:])
                        break
            if len(blog[2].split()) <= 250:
                index_to_be_removed.append(i)

    return [blogs[i] for i in range(0, len(blogs)) if i not in index_to_be_removed]


def get_all_samples(date_returns: dict, word_to_id: dict, bloggers=blogger_names):
    """
    返回训练集和测试集，用id表示
    :return:
    """
    # 连接数据库
    db = pymysql.connect(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
                         port=MYSQL_PORT, charset='utf8')
    cursor = db.cursor()

    # 获取训练集所有数据
    train_blogs = []
    for blogger_name in bloggers:
        sql = "select * from processed_blogs_all_words " \
              "where blogger_name = \'%s\' and created_date between \'%s\' and \'%s\' order by created_date desc;" \
              % (blogger_name, train_start, train_end)
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
        except:
            raise Exception("Error when fetching blogs")
        train_blogs.extend(results)  # 每一个元素是一个tuple，代表了数据库中的一行
    train_blogs = process_blogs(train_blogs)

    # 获取测试集所有数据
    test_blogs = []
    for blogger_name in bloggers:
        sql = "select * from processed_blogs_all_words " \
              "where blogger_name = \'%s\' and created_date between \'%s\' and \'%s\' order by created_date desc;" \
              % (blogger_name, test_start, test_end)
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
        except:
            raise Exception("Error when fetching blogs")
        test_blogs.extend(results)  # 每一个元素是一个tuple，代表了数据库中的一行
    test_blogs = process_blogs(test_blogs)

    # 构建训练集
    x_train: list[list] = []
    y_train: list[int] = []
    for blog in train_blogs:
        words: list[str] = blog[2].split(' ')
        created_date: datetime.datetime = blog[3]
        words_id: list[int] = [word_to_id[word] for word in words if word in word_to_id.keys()]
        # 判断是否大于预设的seq_length，是的话就截取
        if len(words_id) > seq_length:
            mid = int(len(words_id) / 2)
            words_id = words_id[0: int(seq_length / 3)] \
                       + words_id[mid - int(seq_length / 6): mid + int(seq_length / 6)] \
                       + words_id[-int(seq_length / 3):]
        x_train.append(words_id)

        # 根据博文的发布时间，找到对应的标签
        date = created_date.date()
        time = created_date.time()
        # 若博文是在早上9：25前发布的，date推前一天
        if time.__lt__(datetime.time(9, 25, 0, 0)):
            date = date + datetime.timedelta(days=-1)
        # 若date为休市日，推前一天
        while date not in date_returns.keys():
            date = date + datetime.timedelta(days=-1)
        next_three_day_return = date_returns[date]
        if next_three_day_return <= 0:
            y_train.append(0)
        elif next_three_day_return > 0:
            y_train.append(1)

    # 构建测试集
    x_test: list[list] = []
    y_test: list[int] = []
    for blog in test_blogs:
        words: list[str] = blog[2].split(' ')
        created_date: datetime.datetime = blog[3]
        words_id: list[int] = [word_to_id[word] for word in words if word in word_to_id.keys()]
        # 判断是否大于预设的seq_length，是的话就截取
        if len(words_id) > seq_length:
            mid = int(len(words_id) / 2)
            words_id = words_id[0: int(seq_length / 3)] \
                       + words_id[mid - int(seq_length / 6): mid + int(seq_length / 6)] \
                       + words_id[-int(seq_length / 3):]
        x_test.append(words_id)

        # 根据博文的发布时间，找到对应的标签
        date = created_date.date()
        time = created_date.time()
        # 若博文是在早上9：25前发布的，date推前一天
        if time.__lt__(datetime.time(9, 25, 0, 0)):
            date = date + datetime.timedelta(days=-1)
        # 若date为休市日，推前一天
        while date not in date_returns.keys():
            date = date + datetime.timedelta(days=-1)
        next_three_day_return = date_returns[date]
        if next_three_day_return <= 0:
            y_test.append(0)
        elif next_three_day_return > 0:
            y_test.append(1)

    # 将x_train中的元素pad为固定长度max_length
    x_train: numpy.array = keras.preprocessing.sequence.pad_sequences(x_train, seq_length, padding='post')
    # 将y_train中的元素转换为one_hot表示
    y_train: numpy.array = keras.utils.to_categorical(y_train, num_classes=num_classes)

    x_test = keras.preprocessing.sequence.pad_sequences(x_test, seq_length, padding='post')
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

    return x_train, y_train, x_test, y_test


def batch_iter(x, y_, shuffle=True):
    """
    生成器：每迭代一次，生成一个batch的数据
    :param x:
    :param y_:
    :return:
    """
    data_len = len(x)
    num_batch = int(float(data_len - 1) / float(batch_size)) + 1

    # 对数据进行重排
    indices = numpy.random.permutation(numpy.arange(data_len))
    # 注意这里的x的类型为numpy.ndarray，所以才可以这样做
    x_shuffle = x[indices] if shuffle else x
    y_shuffle = y_[indices] if shuffle else y_

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min(start_id + batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


if __name__ == '__main__':
    # dic = get_synonyms('./')
    # stop_words = get_stop_words('./')
    # print(type(stop_words))
    # print(',' in stop_words)
    # print('市场' in dic.keys())
    # print(dic['上升'] == '上涨')
    # for k, v in dic.items():
    #     print(k, v)

    build_vocab()
    # with open('./wordEmbedding.pk', 'rb') as f:
    #     wordEmbedding = pickle.load(f)
    # print(wordEmbedding[0])
    # print(wordEmbedding[1])
    # print(wordEmbedding[4])

    # results = get_all_samples(['余岳桐'])
    # print(results[3])

    # _, word_to_id = read_vocab('./')  # 读取字典
    # date_returns: dict = get_date_returns('./')  # 每个日期接下来三个交易日的return
    # x_train, y_train, x_test, y_test = get_all_samples(date_returns, word_to_id)
    # y_ = numpy.argmax(y_train, 1).tolist()
    # print(len(y_))
    # print(len([label for label in y_ if label == 0]))
    # print(len([label for label in y_ if label == 1]))

    # 计算平均长度
    # train_blogs = get_all_samples(date_returns, word_to_id)
    # print(len(train_blogs))
    # words = [blog[2].split() for blog in train_blogs]
    # length = sum([len(x) for x in words])
    # print(float(length) / float(len(train_blogs)))


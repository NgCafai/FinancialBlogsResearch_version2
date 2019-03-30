import pymysql
from settings import *
# 引入 word2vec
from gensim.models import word2vec
# 引入日志配置
import logging
import gensim


def get_sentences():
    """
    读取数据库中processed_blogs_all_words的内容
    :return:
    """
    # 连接数据库
    db = pymysql.connect(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
                         port=MYSQL_PORT, charset='utf8')
    cursor = db.cursor()

    # 构建查询语句，并获取结果
    sql = "select * from processed_blogs_all_words;"
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
    except:
        print('Error when fetching blogs')
        return

    sentences = []
    for result in results:
        words = result[2].split()
        sentences.append(words)

    return sentences


def train():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # 读取分词内容
    sentences = get_sentences()

    # 构建模型
    model = word2vec.Word2Vec(sentences, size=embedding_dim, min_count=8, workers=4)

    # 保存模型
    model.save('./word2vec_model/model_version1.model')
    model.wv.save_word2vec_format("./word2Vec" + ".bin", binary=True)


if __name__ == '__main__':
    # train()

    model = word2vec.Word2Vec.load('./word2vec_model/model_version1.model')
    print(model.similarity('下杀', '下跌'))
    print(model.most_similar(['上证指数']))
    #
    # wordVec = gensim.models.KeyedVectors.load_word2vec_format("./word2Vec.bin", binary=True)
    # try:
    #     print(wordVec.wv['上涨'])
    #     print(type(wordVec.wv['上涨']))
    # except:
    #     print('no such word')





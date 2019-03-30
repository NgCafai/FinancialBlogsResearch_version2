from settings import *
import jieba
import jieba.analyse
import os
import re
from Util.utils import *


def process_original_blogs_all_words(blogger_name):
    """
    对于某个博主的博文进行处理，包括分词、选择重点词汇、去除停用词、同义词处理，
    然后将所有的分词结果输出到MySQL数据库中
    :param blogger_name:
    :return:
    """
    # 加载自己整理的词典
    jieba.load_userdict("../../Util/MyDict.txt")
    # 连接数据库
    db = pymysql.connect(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
                         port=MYSQL_PORT, charset='utf8')
    cursor = db.cursor()

    # 停用词表
    stop_words = get_stop_words('../../Util/')
    # 同义词字典
    synonyms_dic = get_synonyms('../../Util/')

    # 构建查询语句，并获取结果
    sql = "select * from blogs where blogger_name = \'%s\' order by created_date desc;" % blogger_name
    try:
        cursor.execute(sql)
        blogs = cursor.fetchall()
    except:
        print('Error when fetching blogs')
        return

    # 处理文本并写入MySQL
    for i in range(0, len(blogs)):
        target_sentence = ''
        headline = blogs[i][2]
        if blogger_name in headline:
            headline = headline.replace(blogger_name, '')

        # 跳过无关文章
        if '周末讲堂' in headline or '周末读诗' in headline:
            continue

        target_sentence += headline

        # 正文内容
        content = blogs[i][3]
        # 去掉无用的段落
        if '严重声明' in content:
            content = content[0: content.find('严重声明')]

        paragraphs = content.split('\n')

        # 处理每一段
        for paragraph in paragraphs:
            if paragraph.strip() == '':
                continue

            target_sentence += paragraph

        # 分词并去除停用词
        seg_list = [x.strip() for x in jieba.cut(target_sentence)
                    if x not in stop_words and x.strip() not in stop_words and x.strip() != '']
        # 处理同义词
        for index in range(0, len(seg_list)):
            if seg_list[index] in synonyms_dic.keys():
                seg_list[index] = synonyms_dic[seg_list[index]]

        # 在开头添加博主的名字
        seg_list.insert(0, blogger_name)

        table = 'processed_blogs_all_words'
        keys = 'blogger_name, words, created_date, num_words'
        values = ', '.join(['%s'] * 4)
        sql = 'insert into %s (%s) values (%s)' % (table, keys, values)
        try:
            cursor.execute(sql, (blogger_name, ' '.join(seg_list), blogs[i][-1], len(seg_list)))
            db.commit()
        except:
            print('error when insert: ', blogger_name, blogs[i][-1])
            db.rollback()


if __name__ == '__main__':
    process_original_blogs_all_words("彬哥看盘")

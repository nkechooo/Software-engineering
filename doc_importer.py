import os
import codecs
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError

# 连接 Elasticsearch
es = Elasticsearch(["http://localhost:9200"], verify_certs=False, request_timeout=360)

index_name = "doc_index"
doc_root_dir = "D:\\hw4\\code\\data_doc"  # 文档根目录

# -------- 1. 文件查找函数 --------
def find_files(root_dir, suffixes):
    """
    遍历指定目录，查找具有特定后缀的文件。
    :param root_dir: 根目录
    :param suffixes: 文件后缀列表
    :return: 生成器，返回符合条件的文件路径
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(filename.endswith(suffix) for suffix in suffixes):
                yield os.path.join(dirpath, filename)

# -------- 2. 文本文件解析函数 --------
def parse_txt(file_path):
    """
    解析文本文件内容，提取标题、URL、日期、文件名和类型等信息。
    :param file_path: 文件路径
    :return: 解析后的文档列表
    """
    news_list = []
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    news = {}
    for line in lines:
        line = line.strip()
        if line.startswith("Title:"):
            news['title'] = line.replace("Title:", "").strip()
        elif line.startswith("URL:"):
            news['url'] = line.replace("URL:", "").strip()
        elif line.startswith("Date:"):
            news['date'] = line.replace("Date:", "").strip()
        elif line.startswith("Filename:"):
            news['filename'] = line.replace("Filename:", "").strip()
        elif line.startswith("Type:"):
            news['type'] = line.replace("Type:", "").strip()
        elif line.startswith("----") or line.startswith("—") or line.startswith("-"):
            if news:
                news_list.append(news)
                news = {}
    if news:
        news_list.append(news)
    return news_list

# -------- 3. 数据生成函数 --------
def generate_actions():
    """
    生成 Elasticsearch 的批量导入数据。
    :return: 生成器，返回 Elasticsearch 的 action
    """
    supported_suffixes = (".txt",)
    for doc_path in find_files(doc_root_dir, supported_suffixes):
        print(f"Processing file: {doc_path}")
        try:
            news_list = parse_txt(doc_path)
            print(f"Found {len(news_list)} documents in {doc_path}")
            for news in news_list:
                doc = {
                    "title": news.get("title", ""),
                    "source_file": doc_path,
                    "url": news.get("url", ""),
                    "date": news.get("date", ""),
                    "filename": news.get("filename", ""),
                    "type": news.get("type", "")
                }
                yield {"_index": index_name, "_source": doc}
        except Exception as e:
            print(f"Error processing file '{doc_path}': {e}")

# -------- 4. 批量导入数据 --------
try:
    print("Disabling index refresh interval...")
    es.indices.put_settings(index=index_name, body={"index": {"refresh_interval": "-1"}})

    print("Starting bulk indexing...")
    success, failed = 0, 0
    for ok, action in helpers.streaming_bulk(
        client=es,
        actions=generate_actions(),
        chunk_size=500,
        request_timeout=120
    ):
        if ok:
            success += 1
        else:
            failed += 1

    print(f"Bulk indexing completed: {success} successes, {failed} failures.")
except BulkIndexError as e:
    print(f"BulkIndexError encountered: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    print("Re-enabling index refresh interval...")
    es.indices.put_settings(index=index_name, body={"index": {"refresh_interval": "1s"}})
    print("Data indexing process completed successfully!")
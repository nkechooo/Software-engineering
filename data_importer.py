import os
import codecs
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError
from bs4 import BeautifulSoup

# 连接 Elasticsearch
es = Elasticsearch(["http://localhost:9200"], verify_certs=False, request_timeout=360)

index_name = "doc_index"
txt_root_dir = "D:\hw4\code\data"
snapshot_root_dir = "D:\hw4\code\snapshots_1"
cache_dir = "D:\hw4\code\cache"

os.makedirs(cache_dir, exist_ok=True)

def find_files(root_dir, suffix):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(suffix):
                yield os.path.join(dirpath, filename)

def parse_news_txt(txt_path):
    news_list = []
    with codecs.open(txt_path, 'r', encoding='utf-8') as f:
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
        elif line.startswith("News Snapshot Hash:"):
            news['snapshot_hash'] = line.replace("News Snapshot Hash:", "").strip()
        elif line.startswith("Anchor texts:"):
            anchor_texts = line.replace("Anchor texts:", "").strip()
            anchor_texts = anchor_texts.strip("[]").split(",")  # 去掉括号并分割
            anchor_texts = [text.strip().strip('"') for text in anchor_texts]  # 去掉引号并清理空格
            news['anchor_texts'] = anchor_texts
        elif line.startswith("----") or line.startswith("—") or line.startswith("-"):
            if news:
                news_list.append(news)
                news = {}
    if news:
        news_list.append(news)
    return news_list

# 优化：提前建立快照hash到文件路径的映射
def build_snapshot_map(snapshot_root_dir):
    snapshot_map = {}
    for dirpath, _, filenames in os.walk(snapshot_root_dir):
        for filename in filenames:
            if filename.endswith('.html'):
                hash_val = filename[:-5]  # 去掉 .html
                snapshot_map[hash_val] = os.path.join(dirpath, filename)
    return snapshot_map

snapshot_map = build_snapshot_map(snapshot_root_dir)

def get_snapshot_html_content(snapshot_hash):
    html_path = snapshot_map.get(snapshot_hash)
    if html_path:
        try:
            with codecs.open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
                soup = BeautifulSoup(html_content, 'html.parser')
                return soup.get_text(separator=' ')
        except Exception as e:
            print(f"Error reading snapshot {html_path}: {e}")
    return None

def get_snapshot_html_raw(snapshot_hash):
    html_path = snapshot_map.get(snapshot_hash)
    if html_path:
        try:
            with codecs.open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading raw snapshot {html_path}: {e}")
    return None

def generate_actions():
    seen = set()  # 用于去重
    for txt_path in find_files(txt_root_dir, ".txt"):
        print(f"Processing file: {txt_path}")
        try:
            news_list = parse_news_txt(txt_path)
            print(f"Found {len(news_list)} news in {txt_path}")
            for news in news_list:
                # 用 snapshot_hash作为标识去重
                snapshot_hash = news.get('snapshot_hash', '').strip()
                if not snapshot_hash or snapshot_hash not in snapshot_map:
                    print(f"Warning: snapshot_hash '{snapshot_hash}' not found in snapshot_map, skipping.")
                    continue  # 跳过没有对应html的
                if snapshot_hash in seen:
                    continue  # 跳过重复
                seen.add(snapshot_hash)
                doc = {
                    "title": news.get("title", ""),
                    "url": news.get("url", ""),
                    "date": news.get("date", ""),
                    "snapshot_hash": snapshot_hash,
                    "anchor_texts": news.get("anchor_texts", []),
                    "source_file": txt_path
                }
                snapshot_text = get_snapshot_html_content(snapshot_hash)
                if snapshot_text and snapshot_text.strip():
                    doc["snapshot_text"] = snapshot_text
                    doc["content"] = snapshot_text  # 新增：同步到 content 字段
                else:
                    print(f"Warning: No snapshot_text for hash {snapshot_hash}")
                snapshot_html = get_snapshot_html_raw(snapshot_hash)
                if snapshot_html and snapshot_html.strip():
                    doc["snapshot_html"] = snapshot_html
                else:
                    print(f"Warning: No snapshot_html for hash {snapshot_hash}")
                yield {"_index": index_name, "_source": doc}
        except Exception as e:
            print(f"Error processing file '{txt_path}': {e}")

# 批量导入数据
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
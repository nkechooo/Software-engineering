from elasticsearch import Elasticsearch

# 连接 Elasticsearch（Docker中的Elasticsearch通常使用这个地址）
es = Elasticsearch(["http://localhost:9200"], verify_certs=False, request_timeout=60)

# 设置索引名称
index_name = "doc_index"

# 定义索引的设置与映射
settings = {
    "settings": {
        "analysis": {
            "analyzer": {
                "my_smart_analyzer": {
                    "type": "custom",
                    "tokenizer": "ik_smart",
                    "filter": ["lowercase"]
                },
                "my_max_analyzer": {
                    "type": "custom",
                    "tokenizer": "ik_max_word",
                    "filter": ["lowercase"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "url": {"type": "keyword", "ignore_above": 256},
            "title": {
                "type": "text",
                "analyzer": "my_max_analyzer",
                "index_options": "offsets"
            },
            "content": {
                "type": "text",
                "analyzer": "my_max_analyzer",
                "index_options": "offsets"
            },
            "anchor_texts": {
                "type": "text",
                "analyzer": "my_max_analyzer",
                "index_options": "offsets"
            },
            "date": {"type": "date", "format": "yyyy-MM-dd||yyyy/MM/dd||epoch_millis"},
            "snapshot_hash": {"type": "keyword"},
            "snapshot_text": {
                "type": "text",
                "analyzer": "my_max_analyzer"
            },
            "source_file": {"type": "keyword"},
            "type": {"type": "keyword"},  
            "filename": {"type": "keyword"}  
        }
    }
}

# 删除已有的同名索引（可选）
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

# 创建索引（注意参数要分开传递）
es.indices.create(
    index=index_name,
    settings=settings["settings"],
    mappings=settings["mappings"]
)
print(f"Index '{index_name}' created with given mapping.")
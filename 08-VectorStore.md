# 向量存储的概念指南

**向量存储**是专门设计的数据库，用于通过向量表示（嵌入）来**索引**和**检索**信息。

它们通常用于通过识别语义相似的内容而非依赖精确的关键词匹配，来搜索非结构化数据，如文本、图像和音频。

## 为什么向量存储至关重要

1. **快速高效的搜索**

通过正确存储和索引嵌入向量，向量存储能够快速检索相关信息，即使在处理海量数据集时也能高效工作。

2. **应对数据增长的可扩展性**

随着数据不断扩展，向量存储必须能够高效扩展。一个结构良好的向量存储可以确保系统处理大规模数据时不会出现性能问题，支持无缝增长。

3. **促进语义搜索**

与传统的基于关键词的搜索不同，语义搜索是根据内容的含义来检索信息。向量存储通过查找与用户查询上下文密切相关的段落或部分来实现这一点。这相比于仅能进行精确关键词匹配的原始文本数据库具有明显优势。


## 理解搜索方法

- **基于关键词的搜索**  
  这种方法依赖于通过查询与文档中精确的单词或短语匹配来检索结果。它简单，但无法捕捉单词之间的语义关系。

- **基于相似度的搜索**  
  使用向量表示来评估查询与文档之间的语义相似度。它提供了更准确的结果，尤其是对于自然语言查询。

- **基于分数的相似度搜索**  
  根据查询为每个文档分配一个相似度分数。更高的分数表示相关性更强。常用的度量标准包括 `余弦相似度` 或 `基于距离的评分`。

## 相似度搜索的工作原理

- **嵌入和向量的概念**  
  `嵌入` 是单词或文档在高维空间中的数值表示。它们捕捉了语义信息，从而能够更好地比较查询和文档。

- **相似度度量方法**  
  - **余弦相似度**：衡量两个向量之间的夹角的余弦值。值越接近 1 表示相似度越高。  
  - **欧几里得距离**：计算向量空间中两个点之间的直线距离。距离越小表示相似度越高。

- **评分和排序搜索结果**  
  计算相似度后，文档将被分配一个分数。根据这些分数，结果将按相关性降序排列。

- **搜索算法简要概述**  
  - `TF-IDF`：根据单词在文档中的频率及其在所有文档中的出现频率来赋予单词权重。  
  - `BM25`：`TF-IDF` 的改进版，优化了信息检索的相关性。  
  - `神经搜索`：利用深度学习生成上下文感知的嵌入，以获得更准确的结果。

## 向量存储中的搜索类型

- **相似度搜索**：找到与查询最相似的文档。适用于语义搜索应用。

- **最大边际相关性 (MMR) 搜索**：通过优先选择多样且相关的文档，在搜索结果中平衡相关性和多样性。

- **稀疏检索器**：使用传统的基于关键词的方法，如 `TF-IDF` 或 `BM25` 来检索文档。适用于上下文有限的数据集。

- **密集检索器**：依赖密集的向量嵌入来捕捉语义信息。常见于使用深度学习的现代搜索系统。

- **混合搜索**：结合稀疏和密集的检索方法。通过将密集方法的精确度与稀疏方法的广泛覆盖性结合，达到最优结果。

## 向量存储作为检索器
- **功能**：通过使用 `.as_retriever()` 方法将向量存储转换为检索器，你可以创建一个符合 LangChain 检索器接口的轻量级包装器。这使得你可以使用不同的检索策略，如 `相似度搜索` 和 `最大边际相关性 (MMR) 搜索`，并允许自定义检索参数。  
- **使用案例**：适用于复杂的应用场景，在这些场景中，检索器需要作为更大流程的一部分，例如检索增强生成 (RAG) 系统。它便于与 LangChain 中的其他组件无缝集成，支持诸如集成检索方法和高级查询分析等功能。

总之，虽然直接的向量存储搜索提供了基本的检索能力，但将向量存储转换为检索器，可以提供更强的灵活性和在 LangChain 生态系统中的集成，支持更复杂的检索策略和应用。

## 集成向量数据库


- **Chroma**：一个开源的向量数据库，专为 AI 应用设计，支持高效存储和检索嵌入向量。
- **FAISS**：由 Facebook AI 开发，FAISS（Facebook AI Similarity Search）是一个用于高效相似度搜索和密集向量聚类的库。
- **Pinecone**：一个托管的向量数据库服务，提供高性能的向量相似度搜索，使开发人员能够构建可扩展的 AI 应用程序。
- **Qdrant**：Qdrant（读作 quadrant）是一个向量相似度搜索引擎。它提供了一个生产就绪的服务，具有便捷的 API，用于存储、搜索和管理向量，支持额外的有效载荷和扩展过滤功能。它适用于各种神经网络或基于语义的匹配、分面搜索和其他应用。
- **Elasticsearch**：一个分布式的 RESTful 搜索和分析引擎，支持向量搜索，允许在大型数据集中高效进行相似度搜索。
- **MongoDB**：MongoDB Atlas 向量搜索支持向量嵌入的高效存储、索引和查询，并与操作数据一起使用，方便无缝实现 AI 驱动的应用。
- **pgvector (PostgreSQL)**：一个 PostgreSQL 扩展，添加了向量相似度搜索功能，使得在关系数据库中高效存储和查询向量数据成为可能。
- **Neo4j**：一个图形数据库，存储节点和关系，并原生支持向量搜索，方便执行涉及图形和向量数据的复杂查询。
- **Weaviate**：一个开源的向量数据库，允许存储数据对象和向量嵌入，支持多种数据类型并提供语义搜索功能。
- **Milvus**：一个数据库，专为存储、索引和管理机器学习模型生成的大规模嵌入向量而设计，支持高性能的向量相似度搜索。

这些向量存储在构建需要高效相似度搜索和高维数据管理的应用程序中起着至关重要的作用。

LangChain 提供了一个统一的接口，用于与向量存储交互，使用户可以轻松切换不同的实现方式。

该接口包括用于 **写入**、**删除** 和 **搜索** 向量存储中的文档的核心方法。

主要方法如下：

- `add_documents` : 将一组文本添加到向量存储中。
- `upsert_documents` : 向向量存储中添加新文档，或者如果文档已存在，则更新它们。
  - 在本教程中，我们还将介绍 `upsert_documents_parallel` 方法，它在适用时支持高效的大规模数据处理。
- `delete_documents` : 从向量存储中删除一组文档。
- `similarity_search` : 搜索与给定查询相似的文档。

# Chorma

## Create db


```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
	model="bge-m3",
	base_url='http://localhost:9997/v1',
	api_key='cannot be empty',
	# dimensions=1024,
)

vector_store = Chroma(
	collection_name="langchain_store", 
	embedding_function=embeddings
	)

```

## Add Documents

将一组 Documents 添加到向量存储中, 并生成一组 id, id 可以自己预设置  
返回每个 Documents 的 id

如果 embedding 功能会出现 InternalServerError, 可能文档文本有问题

如果不指定 ID，Chroma 会自动为文档生成唯一 ID，但手动指定 ID 可以帮助你更方便地管理文档。


```python
from langchain_core.documents import Document

texts = [
    "AI helps doctors diagnose diseases faster, improving patient outcomes.",
    "AI can analyze medical images to detect conditions like cancer.",
    "Machine learning predicts patient outcomes based on health data.",
    "AI speeds up drug discovery by predicting the effectiveness of compounds.",
    "AI monitors patients remotely, enabling proactive care for chronic diseases.",
    "AI automates administrative tasks, saving time for healthcare workers.",
    "NLP extracts insights from electronic health records for better care.",
    "AI chatbots help with patient assessments and symptom checking.",
    "AI improves drug manufacturing, ensuring better quality and efficiency.",
    "AI optimizes hospital operations and reduces healthcare costs."
]
# 重复内容会导致报错
# texts = [
#     "AI ",
#     "AI",
#     "Machine learning",
# ]
documents = [
	Document(text, metadata={"source":text})
	for text in texts
]
ids = [f'{i}' for i in range(len(documents))]

ret_ids = vector_store.add_documents(documents=documents, ids=ids)

print(ret_ids)
```

    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


重复添加对于不同向量数据集的处理方式不一样。  
重复添加在 chrome 中不会报错, 应该是直接替代了 id 的原数据, 但换做 Faiss 就会报错, 报错信息显示 id 已存在。


```python
ret_ids = vector_store.add_documents(documents=documents, ids=ids)

print(ret_ids)
```


```python
ret_ids = vector_store.add_documents(documents=documents)

print(ret_ids)
```


```python
from langchain_core.documents import Document

doc1 = Document(page_content="This is a test document.", metadata={"source": "test"})
doc2 = Document(page_content="This is an updated document.", metadata={"source": "test"})

# 文档 ID 可以根据文本的哈希值来生成唯一的 ID
ids = ["doc_1", "doc_2"]

# 第一次插入
vector_store.add_documents(documents=[doc1], ids=["doc_1"])

# 使用 upsert_documents 更新或插入文档
vector_store.upsert_documents(documents=[doc2], ids=["doc_1"])
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[4], line 13
         10 vector_store.add_documents(documents=[doc1], ids=["doc_1"])
         12 # 使用 upsert_documents 更新或插入文档
    ---> 13 vector_store.upsert_documents(documents=[doc2], ids=["doc_1"])


    AttributeError: 'Chroma' object has no attribute 'upsert_documents'


Chroma 没有 upsert_documents 方法, 但有同功能的函数 update_documents, 貌似 add_documents 起到类似的功能(对于一个 id, 更新为新的 Document)


```python
ret_ids = vector_store.update_documents(ids=ids, documents=documents)

print(ret_ids)
```

    None


## Delete Documents by id

chrome 没有 delete_documents 函数, 只能基于 id 删除


```python
ids_4_delete = ["3"]

import traceback
try:
	vector_store.delete(ids=ids_4_delete)
	print(f'ID:{ids_4_delete} deleted')
except Exception as e:
	print(traceback.format_exc())
	print('deleted non-successfully')
```

删除整个数据库


```python
vector_store.delete_collection()
```

## Search

- `similarity_search` 只返回 Documents
- `similarity_search_with_score` 返回 Documents 和对应的 scores
- `similarity_search_by_vector` 可自己转入 embedding vector, 因此可以自己控制 query 的 embedding 与 vecotr database 的 embedding 方法不同. 该函数也有带 scores 返回的版本

similarity_search 方法会基于给定的查询和嵌入向量，在向量存储中查找与之最相似的文档。它通常用于简单的、单次的检索操作。


```python
a_query = "How does AI improve healthcare?"
results = vector_store.similarity_search(
	query=a_query,
	k=3
	)
for index, doc in enumerate(results):
    print(f"[{index}]{doc.page_content} [{doc.metadata}]")
```

    [0]AI monitors patients remotely, enabling proactive care for chronic diseases. [{'source': 'AI monitors patients remotely, enabling proactive care for chronic diseases.'}]
    [1]AI chatbots help with patient assessments and symptom checking. [{'source': 'AI chatbots help with patient assessments and symptom checking.'}]
    [2]AI automates administrative tasks, saving time for healthcare workers. [{'source': 'AI automates administrative tasks, saving time for healthcare workers.'}]



```python
a_query = "How does AI improve healthcare?"
results = vector_store.similarity_search_with_score(
	query=a_query,
	k=3
	)

for index, (doc, score) in enumerate(results):
    print(f"[{index}][SIM={score:3f}]{doc.page_content} [{doc.metadata}]")
```

    [0][SIM=0.718768]AI monitors patients remotely, enabling proactive care for chronic diseases. [{'source': 'AI monitors patients remotely, enabling proactive care for chronic diseases.'}]
    [1][SIM=0.807140]AI chatbots help with patient assessments and symptom checking. [{'source': 'AI chatbots help with patient assessments and symptom checking.'}]
    [2][SIM=0.815210]AI automates administrative tasks, saving time for healthcare workers. [{'source': 'AI automates administrative tasks, saving time for healthcare workers.'}]



```python
query_embedder = OpenAIEmbeddings(
	model="bge-m3",
	base_url='http://localhost:9997/v1',
	api_key='cannot be empty',
	# dimensions=1024,
)

query_vector = query_embedder.embed_query(a_query)
results = db.similarity_search_by_vector(query_vector, k=3)
for index, doc in enumerate(results):
    print(f"[{index}]{doc.page_content} [{doc.metadata}]")

""" 带 score 版本
results = db.similarity_search_with_relevance_scores(query_vector, k=3)
for index, (doc, score) in enumerate(results):
    print(f"[{index}][SIM={score:3f}]{doc.page_content} [{doc.metadata}]")
"""
```

## as_retriever 

as_retriever 方法是 Chroma 向量数据库的一个重要功能，它将 向量数据库 转换为一个 Retriever 对象，这样你就可以在 LangChain 的检索管道中使用它。as_retriever 使得 Chroma 与 LangChain 更好地集成，支持不同的检索策略，并能与其他组件（如问答系统、文档检索系统等）无缝协作。

以下是参数说明
```python
retriever = vector_store.as_retriever(
	# Defines the type of search that the Retriever should perform. Can be “similarity” (default), “mmr”, or “similarity_score_threshold”.
    search_type="mmr",
	# k: num of documents returned
	# fetch_k: Amount of documents to pass to MMR algorithm
	# lambda_mult: Diversity of results returned by MMR; 1 for minimum diversity and 0 for maximum.
    search_kwargs={
		"k": 1, 
		"fetch_k": 2, 
		"lambda_mult": 0.5
		},
)
```



```python
# Retrieve more documents with higher diversity
# Useful if your dataset has many similar documents
vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 6, 'lambda_mult': 0.25}
)

# Fetch more documents for the MMR algorithm to consider
# But only return the top 5
vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 5, 'fetch_k': 50}
)

# Only retrieve documents that have a relevance score
# Above a certain threshold
vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.8}
)

# Only get the single most similar document from the dataset
vector_store.as_retriever(search_kwargs={'k': 1})

# Use a filter to only retrieve documents from a specific paper
vector_store.as_retriever(
    search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
)
```




    VectorStoreRetriever(tags=['FAISS', 'OpenAIEmbeddings'], vectorstore=<langchain_community.vectorstores.faiss.FAISS object at 0x7fe87b72e3f0>, search_kwargs={'filter': {'paper_title': 'GPT-4 Technical Report'}})



最重要的是集成到 langchain 框架, 直接 invoke 调用


```python
retriever = vector_store.as_retriever(search_kwargs={'k': 3})

a_query = "How does AI improve healthcare?"
retriever.invoke(a_query)
```




    [Document(metadata={'source': 'AI monitors patients remotely, enabling proactive care for chronic diseases.'}, page_content='AI monitors patients remotely, enabling proactive care for chronic diseases.'),
     Document(metadata={'source': 'AI chatbots help with patient assessments and symptom checking.'}, page_content='AI chatbots help with patient assessments and symptom checking.'),
     Document(metadata={'source': 'AI automates administrative tasks, saving time for healthcare workers.'}, page_content='AI automates administrative tasks, saving time for healthcare workers.')]



# Faiss

## 初始化


```python
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings

openai_embedding = OpenAIEmbeddings(
	model="bge-m3",
	base_url='http://localhost:9997/v1',
	api_key='cannot be empty',
	# dimensions=1024,
)

embed_dim = len(openai_embedding.embed_query("hello world"))

index = faiss.IndexFlatL2(embed_dim)

vector_store = FAISS(
    embedding_function=openai_embedding,
    index=index,
    docstore= InMemoryDocstore(),
    index_to_docstore_id={}
)
```

## Add Documents


```python
from langchain_core.documents import Document

texts = [
    "AI helps doctors diagnose diseases faster, improving patient outcomes.",
    "AI can analyze medical images to detect conditions like cancer.",
    "Machine learning predicts patient outcomes based on health data.",
    "AI speeds up drug discovery by predicting the effectiveness of compounds.",
    "AI monitors patients remotely, enabling proactive care for chronic diseases.",
    "AI automates administrative tasks, saving time for healthcare workers.",
    "NLP extracts insights from electronic health records for better care.",
    "AI chatbots help with patient assessments and symptom checking.",
    "AI improves drug manufacturing, ensuring better quality and efficiency.",
    "AI optimizes hospital operations and reduces healthcare costs."
]
documents = [
	Document(text)
	for text in texts
]
ids = [f'{i}' for i in range(len(documents))]

ret_ids = vector_store.add_documents(documents=documents, ids=ids)
print(ret_ids)
```

    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


增加重复 Document 会报错
```python
ValueError: Tried to add ids that already exist: {'9', '5', '7', '0', '6', '8', '1', '4', '2', '3'}
```


```python
import traceback
try:
	ret_ids = vector_store.add_documents(documents=documents, ids=ids)
	print(ret_ids)
except Exception as e:
	print(traceback.format_exc())
```

    Traceback (most recent call last):
      File "/tmp/ipykernel_177930/1610519831.py", line 3, in <module>
        ret_ids = vector_store.add_documents(documents=documents, ids=ids)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/site-packages/langchain_core/vectorstores/base.py", line 286, in add_documents
        return self.add_texts(texts, metadatas, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/site-packages/langchain_community/vectorstores/faiss.py", line 341, in add_texts
        return self.__add(texts, embeddings, metadatas=metadatas, ids=ids)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/site-packages/langchain_community/vectorstores/faiss.py", line 316, in __add
        self.docstore.add({id_: doc for id_, doc in zip(ids, documents)})
      File "/data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/site-packages/langchain_community/docstore/in_memory.py", line 28, in add
        raise ValueError(f"Tried to add ids that already exist: {overlapping}")
    ValueError: Tried to add ids that already exist: {'9', '5', '7', '0', '6', '8', '1', '4', '2', '3'}
    


## Get by id

通过 id 返回 Documents. Chroma 没有这个函数


```python
docs = vector_store.get_by_ids(["1","2"])
docs
```




    [Document(id='1', metadata={}, page_content='AI can analyze medical images to detect conditions like cancer.'),
     Document(id='2', metadata={}, page_content='Machine learning predicts patient outcomes based on health data.')]



其他功能和 chroma 类似, 不赘述了

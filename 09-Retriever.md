# VectorStore-backed Retriever

**基于VectorStore的检索器** 是一种文档检索系统，它利用向量存储根据文档的向量表示来进行搜索。这种方法使得基于相似度的搜索变得高效，特别适用于处理非结构化数据。

RAG系统中的文档搜索和响应生成步骤包括：

1. **文档加载**：导入原始文档。
2. **文本切分**：将文本切分成可管理的块。
3. **向量嵌入**：使用嵌入模型将文本转换为数值向量。
4. **存储到向量数据库**：将生成的嵌入向量存储到向量数据库中，以便高效检索。

在查询阶段：
- 流程：用户查询 → 嵌入 → 在向量存储中搜索 → 检索相关块 → LLM生成响应
- 用户的查询被转化为一个嵌入向量，使用嵌入模型。
- 该查询嵌入向量与向量数据库中存储的文档向量进行比较，以 **检索最相关的结果**。
- 检索到的文档块被传递给大语言模型（LLM），该模型基于检索到的信息生成最终响应。


```python
import faiss
from langchain_core.documents import Document
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
```


```python
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
	Document(text, metadata={"source":text})
	for text in texts
]

db = FAISS.from_documents(documents, openai_embedding)
```

一旦向量数据库创建完成，就可以使用检索方法，如 **相似度搜索** 和 **最大边际相关性（MMR）**，加载并查询数据库，从中搜索相关的文本。

`as_retriever` 方法允许你将一个向量数据库转换为一个检索器，从而实现从向量库中高效地搜索和检索文档。

**工作原理**：
* `as_retriever()` 方法将一个向量库（如 FAISS）转换为一个检索器对象，使其与 LangChain 的检索工作流兼容。
* 这个检索器可以直接用于 RAG 流水线，或与大型语言模型（LLM）结合，用于构建智能搜索系统。



```python
retriever = db.as_retriever()
```

## **高级检索器配置**

`as_retriever` 方法允许你配置高级检索策略，如 **相似度搜索**、**最大边际相关性（MMR）** 和 **基于相似度分数阈值的过滤**。

**参数：**

- `**kwargs`：传递给检索函数的关键字参数：
   - `search_type`：指定搜索方法。
     - `"similarity"`：基于余弦相似度返回最相关的文档。
     - `"mmr"`：利用最大边际相关性算法，平衡 **相关性** 和 **多样性**。
     - `"similarity_score_threshold"`：返回相似度分数超过指定阈值的文档。
   - `search_kwargs`：其他用于微调结果的搜索选项：
     - `k`：返回的文档数量（默认值：`4`）。
     - `score_threshold`：用于 `"similarity_score_threshold"` 搜索类型的最小相似度分数（例如：`0.8`）。
     - `fetch_k`：在 MMR 搜索过程中最初检索的文档数量（默认值：`20`）。
     - `lambda_mult`：控制 MMR 结果中的多样性（`0` = 最大多样性，`1` = 最大相关性，默认值：`0.5`）。
     - `filter`：用于选择性文档检索的元数据过滤。

**返回值：**

- `VectorStoreRetriever`：初始化后的检索器对象，可以直接用于文档搜索任务。

**注意事项：**
- 支持多种搜索策略（`similarity`、`MMR`、`similarity_score_threshold`）。
- MMR 通过减少结果中的冗余，提升结果多样性同时保持相关性。
- 元数据过滤使得根据文档属性选择性地检索文档成为可能。
- `tags` 参数可以用于给检索器加标签，以便更好地组织和识别。

**警告：**
- 使用 MMR 时的多样性控制：
  - 小心调整 `fetch_k`（最初检索的文档数量）和 `lambda_mult`（多样性控制因子）以获得最佳平衡。
  - `lambda_mult`：
    - 较低值（< 0.5）→ 优先考虑多样性。
    - 较高值（> 0.5）→ 优先考虑相关性。
  - 为有效的多样性控制，设置 `fetch_k` 大于 `k`。
- 阈值设置：
  - 使用较高的 `score_threshold`（例如 0.95）可能会导致没有结果。
- 元数据过滤：
  - 在应用过滤器之前，确保元数据结构已经定义好。
- 平衡配置：
  - 为了获得最佳的检索性能，保持 `search_type` 和 `search_kwargs` 设置之间的适当平衡。


```python
retriever = db.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={
        "k": 5,  # Return the top 5 most relevant documents
        "score_threshold": 0.5  # Only return documents with a similarity score of 0.4 or higher
    }
)

query = "How does AI improve healthcare?"
results = retriever.invoke(query)

# Display search results
for doc in results:
    print(doc.page_content)
```

    No relevant docs were retrieved using the relevance score threshold 0.5


## 检索器的 `invoke()` 方法

`invoke()` 方法是与检索器交互的主要入口点。它用于根据给定的查询搜索并检索相关的文档。

**工作原理：**
1. **查询提交**：用户提交查询字符串作为输入。
2. **嵌入生成**：如果需要，查询会被转换成向量表示。
3. **搜索过程**：检索器使用指定的搜索策略（如相似度、MMR 等）在向量数据库中进行搜索。
4. **结果返回**：该方法返回一组相关的文档片段。

**参数：**
- `input`（必需）：
   - 用户提供的查询字符串。
   - 查询会被转换成向量，并与存储的文档向量进行相似度比较，以进行基于相似度的检索。

- `config`（可选）：
   - 允许对检索过程进行细粒度控制。
   - 可用于指定 **标签、元数据插入和搜索策略**。

- `**kwargs`（可选）：
   - 允许直接传递 `search_kwargs` 进行高级配置。
   - 示例选项包括：
     - `k`：返回的文档数量。
     - `score_threshold`：文档被包括的最低相似度分数。
     - `fetch_k`：MMR 搜索中最初检索的文档数量。

**返回值：**
- `List[Document]`：
   - 返回包含检索到的文本和元数据的文档对象列表。
   - 每个文档对象包括：
     - `page_content`：文档的主要内容。
     - `metadata`：与文档相关联的元数据（例如，来源、标签）。

**用例 1**


```python
docs = retriever.invoke("What is an embedding?")

for doc in docs:
    print(doc.page_content)
    print("=========================================================")
```

    Machine learning predicts patient outcomes based on health data.
    =========================================================
    AI monitors patients remotely, enabling proactive care for chronic diseases.
    =========================================================
    AI chatbots help with patient assessments and symptom checking.
    =========================================================


**用例 2**


```python
# search options: top 5 results with a similarity score ≥ 0.7
docs = retriever.invoke(
    "What is a vector database?",
    search_kwargs={"k": 5, "score_threshold": 0.7}
)
for doc in docs:
    print(doc.page_content)
    print("=========================================================")
```

    Machine learning predicts patient outcomes based on health data.
    =========================================================
    AI monitors patients remotely, enabling proactive care for chronic diseases.
    =========================================================
    AI chatbots help with patient assessments and symptom checking.
    =========================================================


## 最大边际相关性 (MMR)

**最大边际相关性 (MMR)** 搜索方法是一种文档检索算法，旨在通过平衡相关性和多样性来减少冗余，从而返回结果时提高多样性。

**MMR 的工作原理：**
与仅根据相似度分数返回最相关文档的基本相似度搜索不同，MMR 考虑了两个关键因素：
1. **相关性**：衡量文档与用户查询的匹配程度。
2. **多样性**：确保检索到的文档彼此不同，避免重复的结果。

**关键参数：**
- `search_type="mmr"`：启用 MMR 检索策略。
- `k`：应用多样性过滤后返回的文档数量（默认值：`4`）。
- `fetch_k`：应用多样性过滤前最初检索的文档数量（默认值：`20`）。
- `lambda_mult`：多样性控制因子（`0 = 最大多样性`，`1 = 最大相关性`，默认值：`0.5`）。

## 相似度分数阈值搜索

**相似度分数阈值搜索**是一种检索方法，只有当文档的相似度分数超过预定义的阈值时才会返回。该方法有助于筛选出低相关性的结果，确保返回的文档与查询高度相关。

**关键特性：**
- **相关性过滤**：仅返回相似度分数高于指定阈值的文档。
- **可调精度**：通过 `score_threshold` 参数调整阈值。
- **启用搜索类型**：通过设置 `search_type="similarity_score_threshold"` 启用此搜索方法。

这种搜索方法非常适用于需要**高度精确**结果的任务，例如事实核查或回答技术性查询。

## 配置 `top_k`（调整返回文档的数量）

- 参数 `k` 指定在向量搜索过程中返回的文档数量。它决定了从向量数据库中检索到的 **排名最高**（基于相似度分数）的文档数量。

- 通过在 `search_kwargs` 中设置 `k` 值，可以调整检索到的文档数量。
- 例如，设置 `k=1` 将仅返回 **最相关的 1 篇文档**，该文档基于相似度排序。

# ContextualCompressionRetriever

`ContextualCompressionRetriever` 是 LangChain 中的一种强大工具，旨在通过根据上下文压缩检索到的文档来优化检索过程。这个检索器特别适用于需要对大量数据进行动态总结或过滤的场景，确保只有最相关的信息传递到后续处理步骤。

`ContextualCompressionRetriever` 的主要特点包括：

- **上下文感知压缩**：文档会根据特定的上下文或查询进行压缩，确保相关性并减少冗余。
- **灵活的集成**：与其他 LangChain 组件无缝工作，便于集成到现有的管道中。
- **可定制的压缩**：支持使用不同的压缩技术，包括摘要模型和基于嵌入的方法，来根据需求定制检索过程。

`ContextualCompressionRetriever` 特别适用于以下应用：

- 为问答系统总结大量数据。
- 通过提供简洁且相关的回答来提升聊天机器人性能。
- 提高文档密集型任务（如法律分析或学术研究）的效率。

通过使用这个检索器，开发者可以显著减少计算开销，并提高提供给最终用户的信息质量。


```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# 1. Generate Loader to lthe text file using TextLoader
loader = TextLoader("./data/appendix-keywords.txt")\

# 2. Generate text chunks using CharacterTextSplitter and split the text into chunks of 300 characters with no overlap.
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
texts = loader.load_and_split(text_splitter)

# 3. Generate vector store using FAISS and convert it to retriever
embedder = OpenAIEmbeddings(
	model="bge-m3",
	base_url='http://localhost:9997/v1',
	api_key='cannot be empty',
	# dimensions=1024,
)
retriever = FAISS.from_documents(texts, embedder).as_retriever(search_kwargs={"k": 10})

# 4. Query the retriever to find relevant documents
docs = retriever.invoke("What is the definition of Multimodal?")

# 5. Print the relevant documents
for i, d in enumerate(docs):
	print(f"document {i+1}:\n\n" + d.page_content)
```

    Created a chunk of size 419, which is longer than the specified 400


    document 1:
    
    Semantic Search
    document 2:
    
    Definition: Semantic search is a search method that goes beyond simple keyword matching by understanding the meaning of the user’s query to return relevant results.
    Example: If a user searches for “planets in the solar system,” the system might return information about related planets such as “Jupiter” or “Mars.”
    Related Keywords: Natural Language Processing, Search Algorithms, Data Mining
    document 3:
    
    Definition: A token refers to a smaller unit of text obtained by splitting a larger text. It can be a word, sentence, or phrase.
    Example: The sentence “I go to school” can be split into tokens: “I”, “go”, “to”, “school”.
    Related Keywords: Tokenization, Natural Language Processing, Parsing
    
    Tokenizer
    document 4:
    
    Definition: A tokenizer is a tool that splits text data into tokens. It is commonly used in natural language processing for data preprocessing.
    Example: The sentence “I love programming.” can be tokenized into [“I”, “love”, “programming”, “.”].
    Related Keywords: Tokenization, Natural Language Processing, Parsing
    
    VectorStore
    document 5:
    
    Definition: A vector store is a system for storing data in vector form. It is used for tasks like retrieval, classification, and other data analysis.
    Example: Word embedding vectors can be stored in a database for quick access.
    Related Keywords: Embedding, Database, Vectorization
    
    SQL
    document 6:
    
    Definition: SQL (Structured Query Language) is a programming language for managing data in databases. It supports operations like querying, modifying, inserting, and deleting data.
    Example: SELECT * FROM users WHERE age > 18; retrieves information about users older than 18.
    Related Keywords: Database, Query, Data Management
    
    CSV
    document 7:
    
    Definition: CSV (Comma-Separated Values) is a file format for storing data where each value is separated by a comma. It is often used for simple data storage and exchange in tabular form.
    Example: A CSV file with headers “Name, Age, Job” might contain data like “John Doe, 30, Developer”.
    Related Keywords: File Format, Data Handling, Data Exchange
    
    JSON
    document 8:
    
    Definition: JSON (JavaScript Object Notation) is a lightweight data exchange format that represents data objects in a human- and machine-readable text format.
    Example: {"name": "John Doe", "age": 30, "job": "Developer"} is an example of JSON data.
    Related Keywords: Data Exchange, Web Development, API
    
    Transformer
    document 9:
    
    Definition: A transformer is a type of deep learning model used in natural language processing for tasks like translation, summarization, and text generation. It is based on the attention mechanism.
    Example: Google Translate uses transformer models to perform translations between languages.
    Related Keywords: Deep Learning, Natural Language Processing, Attention
    
    HuggingFace
    document 10:
    
    Definition: HuggingFace is a library that provides pre-trained models and tools for natural language processing, making NLP tasks more accessible to researchers and developers.
    Example: HuggingFace’s Transformers library can be used for tasks like sentiment analysis and text generation.
    Related Keywords: Natural Language Processing, Deep Learning, Library
    
    Digital Transformation


使用 `LLMChainExtractor` 创建的 `DocumentCompressor` 正是应用于检索器的，即 `ContextualCompressionRetriever`。

`ContextualCompressionRetriever` 会通过去除无关信息并专注于最相关的信息来压缩文档。

### LLMChainFilter

`LLMChainFilter` 是一个简单但强大的压缩器，它使用 LLM 链来决定从最初检索到的文档中哪些应该被过滤，哪些应该被返回。


```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

# Before applying ContextualCompressionRetriever
docs = retriever.invoke("What is the definition of Multimodal?")
for i, d in enumerate(docs):
	print(f"document {i+1}:\n\n" + d.page_content)
print("="*62)
print("="*15 + "After applying LLMChainExtractor" + "="*15)


# After applying ContextualCompressionRetriever
# 1. Generate LLM
llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)


# 2. Generate compressor using LLMChainExtractor
compressor = LLMChainExtractor.from_llm(llm)

# 3. Generate compression retriever using ContextualCompressionRetriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)

# 4. Query the compression retriever to find relevant documents
compressed_docs = (
    compression_retriever.invoke( 
        "What is the definition of Multimodal?"
    )
)

# 5. Print the relevant documents
for i, d in enumerate(compressed_docs):
	print(f"document {i+1}:\n\n" + d.page_content)
```

    document 1:
    
    Semantic Search
    document 2:
    
    Definition: Semantic search is a search method that goes beyond simple keyword matching by understanding the meaning of the user’s query to return relevant results.
    Example: If a user searches for “planets in the solar system,” the system might return information about related planets such as “Jupiter” or “Mars.”
    Related Keywords: Natural Language Processing, Search Algorithms, Data Mining
    document 3:
    
    Definition: A token refers to a smaller unit of text obtained by splitting a larger text. It can be a word, sentence, or phrase.
    Example: The sentence “I go to school” can be split into tokens: “I”, “go”, “to”, “school”.
    Related Keywords: Tokenization, Natural Language Processing, Parsing
    
    Tokenizer
    document 4:
    
    Definition: A tokenizer is a tool that splits text data into tokens. It is commonly used in natural language processing for data preprocessing.
    Example: The sentence “I love programming.” can be tokenized into [“I”, “love”, “programming”, “.”].
    Related Keywords: Tokenization, Natural Language Processing, Parsing
    
    VectorStore
    document 5:
    
    Definition: A vector store is a system for storing data in vector form. It is used for tasks like retrieval, classification, and other data analysis.
    Example: Word embedding vectors can be stored in a database for quick access.
    Related Keywords: Embedding, Database, Vectorization
    
    SQL
    document 6:
    
    Definition: SQL (Structured Query Language) is a programming language for managing data in databases. It supports operations like querying, modifying, inserting, and deleting data.
    Example: SELECT * FROM users WHERE age > 18; retrieves information about users older than 18.
    Related Keywords: Database, Query, Data Management
    
    CSV
    document 7:
    
    Definition: CSV (Comma-Separated Values) is a file format for storing data where each value is separated by a comma. It is often used for simple data storage and exchange in tabular form.
    Example: A CSV file with headers “Name, Age, Job” might contain data like “John Doe, 30, Developer”.
    Related Keywords: File Format, Data Handling, Data Exchange
    
    JSON
    document 8:
    
    Definition: JSON (JavaScript Object Notation) is a lightweight data exchange format that represents data objects in a human- and machine-readable text format.
    Example: {"name": "John Doe", "age": 30, "job": "Developer"} is an example of JSON data.
    Related Keywords: Data Exchange, Web Development, API
    
    Transformer
    document 9:
    
    Definition: A transformer is a type of deep learning model used in natural language processing for tasks like translation, summarization, and text generation. It is based on the attention mechanism.
    Example: Google Translate uses transformer models to perform translations between languages.
    Related Keywords: Deep Learning, Natural Language Processing, Attention
    
    HuggingFace
    document 10:
    
    Definition: HuggingFace is a library that provides pre-trained models and tools for natural language processing, making NLP tasks more accessible to researchers and developers.
    Example: HuggingFace’s Transformers library can be used for tasks like sentiment analysis and text generation.
    Related Keywords: Natural Language Processing, Deep Learning, Library
    
    Digital Transformation
    ==============================================================
    ===============After applying LLMChainExtractor===============


大模型把无关内容都过滤了, 虽然我 embedding 很拉, 没能抽到相关内容  
以下是一个过滤效果的展示, 把定义成功保留, 示例被过滤掉


```python
text = \
"""
Multimodal
Definition: Multimodal refers to the technology that combines multiple types of data modes (e.g., text, images, sound) to process and extract richer and more accurate information or predictions.
Example: A system that analyzes both images and descriptive text to perform more accurate image classification is an example of multimodal technology.
Relate
"""
docs = [Document(text)]
query = "What is the definition of Multimodal?"
compressed_docs = compressor.compress_documents(docs, query)
print(compressed_docs[0].page_content)
```

    Multimodal
    Definition: Multimodal refers to the technology that combines multiple types of data modes (e.g., text, images, sound) to process and extract richer and more accurate information or predictions.


源码分析

这是 `ContextualCompressionRetriever` 的检索函数 `_get_relevant_documents`的关键代码:
```python
	docs = self.base_retriever.invoke(
		query, config={"callbacks": run_manager.get_child()}, **kwargs
	)
	if docs:
		compressed_docs = self.base_compressor.compress_documents(
			docs, query, callbacks=run_manager.get_child()
		)
		return list(compressed_docs)
	else:
		return []
```
首先还是 base_retriever 支持返回检索结果, 再接过 base_compressor 压缩

这是 base_compresser 类 `LLMChainExtractor` 的 `compress_documents`函数关键部分:
```python
	compressed_docs = []
	for doc in documents:
		_input = self.get_input(query, doc) # 产生 {"question": query, "context": doc.page_content}
		output_ = self.llm_chain.invoke(_input, config={"callbacks": callbacks}) # 调用大模型抽取内容
		if isinstance(self.llm_chain, LLMChain):
			output = output_[self.llm_chain.output_key]
			if self.llm_chain.prompt.output_parser is not None:
				output = self.llm_chain.prompt.output_parser.parse(output)
		else:
			output = output_
		if len(output) == 0:
			continue
		compressed_docs.append(
			Document(page_content=cast(str, output), metadata=doc.metadata)
		)
	return compressed_docs
```

这是调用大模型抽取内容的 prompt 模板
```python
"""
Given the following question and context, extract any part of the context *AS IS* that is relevant to answer the question. If none of the context is relevant return {no_output_str}. 

Remember, *DO NOT* edit the extracted parts of the context.

> Question: {{question}}
> Context:
>>>
{{context}}
>>>
Extracted relevant parts:
"""
```

### EmbeddingsFilter

对每个检索到的文档执行额外的 LLM 调用既昂贵又缓慢。  
`EmbeddingsFilter` 提供了一个更经济且更快速的选项，通过嵌入文档和查询，只返回那些与查询的嵌入相似度足够高的文档。

这种方法在保持搜索结果相关性的同时，节省了计算成本和时间。  
该过程涉及使用 `EmbeddingsFilter` 和 `ContextualCompressionRetriever` 压缩并检索相关文档。

- `EmbeddingsFilter` 用于过滤超过指定相似度阈值（0.86）的文档。


```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings

# 1. Generate embeddings using OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
	model="bge-m3",
	base_url='http://localhost:9997/v1',
	api_key='cannot be empty',
	# dimensions=1024,
)

# 2. Generate EmbedingsFilter object that has similarity threshold of 0.86
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.86)

# 3. Generate ContextualCompressionRetriever object using EmbeddingsFilter and retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, 
    base_retriever=retriever
)

# 4. Query the compression retriever to find relevant documents
compressed_docs = compression_retriever.invoke(
    "What is the definition of Multimodal?"
)

# 5. Print the relevant documents
for i, d in enumerate(compressed_docs):
	print(f"document {i+1}:\n\n" + d.page_content)
```

这个方法也只是将 base_retriever 的返回结果经过 EmbeddingsFilter 的相似度阈值过滤, 可以选择更强的 embedding model 来强化相似度准确度

# Ensemble Retriever 多路召回

`EnsembleRetriever` 集成了稀疏和密集检索算法的优点，通过使用权重和运行时配置来定制性能。

**关键特点**
1. 集成多个检索器：接受不同类型的检索器作为输入并结合结果。
2. 结果重新排序：使用[倒排排名融合](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)算法重新排序结果。
3. 混合检索：主要使用`稀疏检索器`（例如 BM25）和`密集检索器`（例如 嵌入相似度）相结合。

**优势**
- 稀疏检索器：有效进行基于关键词的检索。
- 密集检索器：有效进行基于语义相似度的检索。

由于这些互补特性，`EnsembleRetriever` 可以在各种检索场景中提供更好的性能。


```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# list sample documents
doc_list = [
    "I like apples",
    "I like apple company",
    "I like apple's iphone",
    "Apple is my favorite company",
    "I like apple's ipad",
    "I like apple's macbook",
]

# Initialize the bm25 retriever and faiss retriever.
bm25_retriever = BM25Retriever.from_texts(
    doc_list,
)
bm25_retriever.k = 2  # Set the number of search results for BM25Retriever to 1.

embedding = OpenAIEmbeddings(
	model="bge-m3",
	base_url='http://localhost:9997/v1',
	api_key='cannot be empty',
	# dimensions=1024,
	)

faiss_vectorstore = FAISS.from_texts(
    doc_list,
    embedding,
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

# Initialize the ensemble retriever.
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.7, 0.3],
)
```


```python
# Get the search results document.
query = "my favorite fruit is apple"
ensemble_result = ensemble_retriever.invoke(query)
bm25_result = bm25_retriever.invoke(query)
faiss_result = faiss_retriever.invoke(query)

# Output the fetched documents.
print("[Ensemble Retriever]")
for doc in ensemble_result:
    print(f"Content: {doc.page_content}")
    print()

print("[BM25 Retriever]")
for doc in bm25_result:
    print(f"Content: {doc.page_content}")
    print()

print("[FAISS Retriever]")
for doc in faiss_result:
    print(f"Content: {doc.page_content}")
    print()
```

    [Ensemble Retriever]
    Content: Apple is my favorite company
    
    Content: I like apple company
    
    Content: I like apples
    
    [BM25 Retriever]
    Content: Apple is my favorite company
    
    Content: I like apple company
    
    [FAISS Retriever]
    Content: Apple is my favorite company
    
    Content: I like apples
    


源码分析

`EnsembleRetriever` 的 `rank_fusion` 函数:
```python
retriever_docs = [
	retriever.invoke(
		query,
		patch_config(
			config, callbacks=run_manager.get_child(tag=f"retriever_{i + 1}")
		),
	)
	for i, retriever in enumerate(self.retrievers)
]

# Enforce that retrieved docs are Documents for each list in retriever_docs
for i in range(len(retriever_docs)):
	retriever_docs[i] = [
		Document(page_content=cast(str, doc)) if isinstance(doc, str) else doc
		for doc in retriever_docs[i]
	]

# apply rank fusion
fused_documents = self.weighted_reciprocal_rank(retriever_docs)
```

每个 retriever 单独调用, 返回多组 Documents, 再经过 `weighted_reciprocal_rank`:

```python
rrf_score: Dict[str, float] = defaultdict(float)
for doc_list, weight in zip(doc_lists, self.weights):
	for rank, doc in enumerate(doc_list, start=1):
		rrf_score[
			(
				doc.page_content
				if self.id_key is None
				else doc.metadata[self.id_key]
			)
		] += weight / (rank + self.c)

# Docs are deduplicated by their contents then sorted by their scores
all_docs = chain.from_iterable(doc_lists)
sorted_docs = sorted(
	unique_by_key(
		all_docs,
		lambda doc: (
			doc.page_content
			if self.id_key is None
			else doc.metadata[self.id_key]
		),
	),
	reverse=True,
	key=lambda doc: rrf_score[
		doc.page_content if self.id_key is None else doc.metadata[self.id_key]
	],
)
```
基于 weights 对 Documents 重排序

# Long Context Reorder

无论模型的架构如何，当检索的文档超过 10 个时，性能都会显著下降。

简单来说，当模型需要在长上下文的中间部分访问相关信息时，它往往会忽视提供的文档。

更多细节，请参阅以下论文：

- [https://arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172)

为了避免这个问题，您可以在检索后重新排序文档，从而防止性能下降。

可以创建一个检索器，它使用 Chroma 向量数据库存储和搜索文本数据。然后，使用检索器的 `invoke` 方法，针对给定的查询搜索出高度相关的文档。


```python
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Get embeddings
embeddings = OpenAIEmbeddings(
	model="bge-m3",
	base_url='http://localhost:9997/v1',
	api_key='cannot be empty',
	# dimensions=1024,
	)

texts = [
    "This is just a random text I wrote.",
    "ChatGPT, an AI designed to converse with users, can answer various questions.",
    "iPhone, iPad, MacBook are representative products released by Apple.",
    "ChatGPT was developed by OpenAI and is continuously being improved.",
    "ChatGPT has learned from vast amounts of data to understand user questions and generate appropriate answers.",
    "Wearable devices like Apple Watch and AirPods are also part of Apple's popular product line.",
    "ChatGPT can be used to solve complex problems or suggest creative ideas.",
    "Bitcoin is also called digital gold and is gaining popularity as a store of value.",
    "ChatGPT's capabilities are continuously evolving through ongoing learning and updates.",
    "The FIFA World Cup is held every four years and is the biggest event in international football.",
]



# Create a retriever (Set K to 10)
retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
    search_kwargs={"k": 10}
)
```


```python
query = "What can you tell me about ChatGPT?"

# Retrieves relevant documents sorted by relevance score.
docs = retriever.invoke(query)
docs
```




    [Document(metadata={}, page_content='Bitcoin is also called digital gold and is gaining popularity as a store of value.'),
     Document(metadata={}, page_content='The FIFA World Cup is held every four years and is the biggest event in international football.'),
     Document(metadata={}, page_content="Wearable devices like Apple Watch and AirPods are also part of Apple's popular product line."),
     Document(metadata={}, page_content='iPhone, iPad, MacBook are representative products released by Apple.'),
     Document(metadata={}, page_content='This is just a random text I wrote.'),
     Document(metadata={}, page_content='ChatGPT, an AI designed to converse with users, can answer various questions.'),
     Document(metadata={}, page_content='ChatGPT was developed by OpenAI and is continuously being improved.'),
     Document(metadata={}, page_content='ChatGPT has learned from vast amounts of data to understand user questions and generate appropriate answers.'),
     Document(metadata={}, page_content='ChatGPT can be used to solve complex problems or suggest creative ideas.'),
     Document(metadata={}, page_content="ChatGPT's capabilities are continuously evolving through ongoing learning and updates.")]



创建一个 `LongContextReorder` 类的实例。

- 调用 `reordering.transform_documents(docs)` 来重新排序文档列表。
- 相关性较低的文档会被置于列表的中间，而相关性较高的文档会被放置在列表的开头和结尾。


```python
from langchain_community.document_transformers import LongContextReorder
reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)

reordered_docs
```




    [Document(metadata={}, page_content='The FIFA World Cup is held every four years and is the biggest event in international football.'),
     Document(metadata={}, page_content='iPhone, iPad, MacBook are representative products released by Apple.'),
     Document(metadata={}, page_content='ChatGPT, an AI designed to converse with users, can answer various questions.'),
     Document(metadata={}, page_content='ChatGPT has learned from vast amounts of data to understand user questions and generate appropriate answers.'),
     Document(metadata={}, page_content="ChatGPT's capabilities are continuously evolving through ongoing learning and updates."),
     Document(metadata={}, page_content='ChatGPT can be used to solve complex problems or suggest creative ideas.'),
     Document(metadata={}, page_content='ChatGPT was developed by OpenAI and is continuously being improved.'),
     Document(metadata={}, page_content='This is just a random text I wrote.'),
     Document(metadata={}, page_content="Wearable devices like Apple Watch and AirPods are also part of Apple's popular product line."),
     Document(metadata={}, page_content='Bitcoin is also called digital gold and is gaining popularity as a store of value.')]



源码分析

```python
documents.reverse()
reordered_result = []
for i, value in enumerate(documents):
	if i % 2 == 1:
		reordered_result.append(value)
	else:
		reordered_result.insert(0, value)
```
原顺序是相似度由高到低的, 他只是在原顺序的基础上把高相似度的放散在头部和尾部, 低相关的放在中部.  

> 当模型需要在长上下文的中间部分访问相关信息时，它往往会忽视提供的文档

按这种说法, 模型会更注重头部和尾部的文档

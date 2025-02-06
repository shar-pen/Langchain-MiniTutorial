# Reranker

目的:  
- 对检索到的文档进行重排序，优化其排名，优先展示与查询最相关的结果。

结构:  
- 接受**查询**和**文档**作为单一输入对，进行联合处理。

机制:  
- **单一输入对**：  
  将**查询**和**文档**作为组合输入，直接输出相关性评分。
- **自注意力机制**：  
  使用自注意力机制联合分析**查询**和**文档**，有效捕捉它们之间的语义关系。

优势:  
- **更高的准确性**：  
  提供更精确的相似度评分。
- **深度语境分析**：  
  探索**查询**和**文档**之间的语义细微差别。

局限性:  
- **高计算成本**：  
  处理可能会消耗较多时间。
- **扩展性问题**：  
  未经过优化时，不适用于大规模文档集合。

实际应用:  
- **双编码器（Bi-Encoder）**通过计算轻量级的相似度分数快速检索候选**文档**。
- **交叉编码器（Cross Encoder）**通过深入分析**查询**与检索到的**文档**之间的语义关系，进一步优化这些结果。


```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load documents
documents = TextLoader("./data/appendix-keywords.txt").load()

# Configure text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Split documents into chunks
texts = text_splitter.split_documents(documents)

# # Set up the embedding model
embeddings_model = OpenAIEmbeddings(
	model="bge-m3",
	base_url='http://localhost:9997/v1',
	api_key='cannot be empty',
	# dimensions=1024,
)

# Create FAISS index from documents and set up retriever
retriever = FAISS.from_documents(texts, embeddings_model).as_retriever(
    search_kwargs={"k": 30}
)

# Define the query
query = "Can you tell me about Word2Vec?"

# Execute the query and retrieve results
docs = retriever.invoke(query)

# Display the retrieved documents
for i, d in enumerate(docs):
	print(f"document {i+1}:\n\n" + d.page_content)
	print('-' * 100)

```

使用 `ContextualCompressionRetriever` 来包装 `base_retriever`。`CrossEncoderReranker` 利用 `HuggingFaceCrossEncoder` 对检索到的结果进行重新排序。

之前 `ContextualCompressionRetriever` 只是对文档压缩, 现在起到过滤作用


```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Initialize the model
model = HuggingFaceCrossEncoder(
	model_name="../DataCollection/officials/bge-reranker-v2-m3",
	model_kwargs = {'device': 'cuda:6'}
	)

# Select the top 3 documents
compressor = CrossEncoderReranker(model=model, top_n=3)

# Initialize the contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# Retrieve compressed documents
compressed_docs = compression_retriever.invoke("Can you tell me about Word2Vec?")

# Display the documents
for i, d in enumerate(docs):
	print(f"document {i+1}:\n\n" + d.page_content)
	print('-' * 100)
```

我没有在 langchain 上找到基于 api 的 reranker 类

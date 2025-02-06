# 理解 RAG 的基本结构

RAG（检索增强生成）流程中的预处理阶段涉及四个步骤，用于加载、拆分、嵌入和存储文档到向量数据库（Vector DB）。

1. 预处理 ## 步骤 1 到 4
- **步骤 1：文档加载**：加载文档内容。
- **步骤 2：文本拆分**：根据特定标准将文档拆分成多个块。
- **步骤 3：嵌入**：为拆分后的文档块生成嵌入，并为存储做准备。
- **步骤 4：向量数据库存储**：将生成的嵌入存储到向量数据库中。

以上可以称为 Indexing。一个收集数据并对其进行索引的管道。这个过程通常是在离线进行的。

2. RAG 执行（运行时） - 步骤 5 到 8
- **步骤 5：检索器**：定义一个检索器，根据输入查询从数据库中获取结果。检索器使用搜索算法，通常分为**密集型**和**稀疏型**：
  - **密集型**：基于相似度的检索。
  - **稀疏型**：基于关键词的检索。
- **步骤 6：提示生成**：为执行 RAG 创建一个提示。提示中的`context`包含从文档中检索到的内容。通过提示工程，可以指定回答的格式。
- **步骤 7：大语言模型（LLM）**：定义使用的大语言模型（例如 GPT-3.5、GPT-4 或 Claude）。
- **步骤 8：链式连接**：创建一个链，将提示、大语言模型和输出连接起来。

以上可称为 Retrieval and Generation 。实际的RAG链实时处理用户查询，从索引中检索相关数据，并将其传递给模型。

# RAG 基本 pipeline

以下是理解 RAG（检索增强生成）基本结构的框架代码。  
每个模块的内容可以根据具体场景进行调整，从而允许逐步改进结构以适应文档。  
（在每个步骤中，可以应用不同的选项或新技术。）


```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
```

## **步骤 1：文档加载**：加载文档内容。


```python
# Step 1: Load Documents
loader = TextLoader("data/appendix-keywords.txt", encoding="utf-8")
docs = loader.load()
print(f"Number of pages in the document: {len(docs)}")
```

    Number of pages in the document: 1


## **步骤 2：文本拆分**：根据特定标准将文档拆分成多个块。


```python
# Step 2: Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"Number of split chunks: {len(split_documents)}")
```

    Number of split chunks: 30


## **步骤 3：嵌入**：为拆分后的文档块生成嵌入，并为存储做准备。



```python
# Step 3: Generate Embeddings
embeddings = OpenAIEmbeddings(
	model="bge-m3",
	base_url='http://localhost:9997/v1',
	api_key='cannot be empty',
	# dimensions=1024,
)
```

## **步骤 4：向量数据库存储**：将生成的嵌入存储到向量数据库中。



```python
# Step 4: Create and Save the Database
# Create a vector store.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
```


```python
# test similarity_search
for doc in vectorstore.similarity_search("URBAN MOBILITY", k=1):
    print(doc.page_content)
```


## **步骤 5：检索器**：定义一个检索器，根据输入查询从数据库中获取结果。检索器使用搜索算法，通常分为**密集型**和**稀疏型**：



```python
# Step 5: Create Retriever
# Search and retrieve information contained in the documents.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 1})
```


```python
retriever.invoke("What is the phased implementation timeline for the EU AI Act?")
```


## **步骤 6：提示生成**：为执行 RAG 创建一个提示。提示中的`context`包含从文档中检索到的内容。通过提示工程，可以指定回答的格式。



```python
# Step 6: Create Prompt
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 

#Context: 
{context}

#Question:
{question}

#Answer:"""
)
prompt
```




    PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template="You are an assistant for question-answering tasks. \nUse the following pieces of retrieved context to answer the question. \nIf you don't know the answer, just say that you don't know. \n\n#Context: \n{context}\n\n#Question:\n{question}\n\n#Answer:")



## **步骤 7：大语言模型（LLM）**：定义使用的大语言模型（例如 GPT-3.5、GPT-4 或 Claude）。



```python
# Step 7: Load LLM
llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)
```

## **步骤 8：链式连接**：创建一个链，将提示、大语言模型和输出连接起来。


```python
# Step 8: Create Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```


```python
# Run Chain
# Input a query about the document and print the response.
question = "Where has the application of AI in healthcare been confined to so far?"
response = chain.invoke(question)
print(response)
```

# 完整版本


```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Step 1: Load Documents
loader = TextLoader("data/appendix-keywords.txt", encoding="utf-8")
docs = loader.load()
print(f"Number of pages in the document: {len(docs)}")

# Step 2: Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"Number of split chunks: {len(split_documents)}")

# Step 3: Generate Embeddings
embeddings = OpenAIEmbeddings(
	model="bge-m3",
	base_url='http://localhost:9997/v1',
	api_key='cannot be empty',
	# dimensions=1024,
)

# Step 4: Create and Save the Database
# Create a vector store.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# Step 5: Create Retriever
# Search and retrieve information contained in the documents.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 1})

# Step 6: Create Prompt
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 

#Context: 
{context}

#Question:
{question}

#Answer:"""
)

# Step 7: Load LLM
llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)

# Step 8: Create Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run Chain
# Input a query about the document and print the response.
question = "Where has the application of AI in healthcare been confined to so far?"
response = chain.invoke(question)
print(response)
```

# 多轮对话

创建一个记住之前对话的链条, 将对话历史添加到 chain

- 使用 `MessagesPlaceholder` 来包含对话历史。
- 定义一个提示语，接收用户输入的问题。
- 创建一个 `ChatOpenAI` 实例，使用 OpenAI 的 `ChatGPT` 模型。
- 通过将提示语、语言模型和输出解析器连接起来，构建链条。
- 使用 `StrOutputParser` 将模型的输出转换为字符串。

## 基础问答框架


```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# Defining the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Question-Answering chatbot. Please provide an answer to the given question.",
        ),
        # Please use the key 'chat_history' for conversation history without changing it if possible!
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "#Question:\n{question}"),  # Use user input as a variable
    ]
)

# Generating an LLM
llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)

# Creating a regular Chain
chain = prompt | llm | StrOutputParser()
```

## 创建可管理对话历史的 chain

创建一个记录对话的链条 (chain_with_history)

- 创建一个字典来存储会话记录。
- 定义一个函数，根据会话 ID 检索会话记录。如果会话 ID 不在存储中，创建一个新的 `ChatMessageHistory` 对象。
- 创建一个 `RunnableWithMessageHistory` 对象来管理对话历史。


```python
# Dictionary to store session records
store = {}

# Function to retrieve session records based on session ID
def get_session_history(session_ids):
    print(f"[Conversation Session ID]: {session_ids}")
    if session_ids not in store:  # If the session ID is not in the store
        # Create a new ChatMessageHistory object and save it to the store
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # Return the session history for the corresponding session ID


chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # Function to retrieve session history
    input_messages_key="question",  # Key for the template variable that will contain the user's question
    history_messages_key="chat_history",  # Key for the history messages
)
```

## 多轮 QA 测试


```python
chain_with_history.invoke(
    # Input question
    {"question": "My name is Jack."},
    # Record the conversation based on the session ID.
    config={"configurable": {"session_id": "abc123"}},
)
```

    [Conversation Session ID]: abc123





    'Hello Jack! Nice to meet you. How can I assist you today?'




```python
chain_with_history.invoke(
    # Input question
    {"question": "What is my name?"},
    # Record the conversation based on the session ID.
    config={"configurable": {"session_id": "abc123"}},
)
```

    [Conversation Session ID]: abc123





    'Your name is Jack.'




```python
print(store['abc123'])
```

    Human: My name is Jack.
    AI: Hello Jack! Nice to meet you. How can I assist you today?
    Human: What is my name?
    AI: Your name is Jack.


# 带多轮对话的 RAG 框架

## 基础问答框架


```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from operator import itemgetter

# Step 1: Load Documents
loader = PDFPlumberLoader("data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf") 
docs = loader.load()

# Step 2: Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"Number of split chunks: {len(split_documents)}")

# Step 3: Generate Embeddings
embeddings = OpenAIEmbeddings(
	model="bge-m3",
	base_url='http://localhost:9997/v1',
	api_key='cannot be empty',
	# dimensions=1024,
)

# Step 4: Create and Save the Database
# Create a vector store.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# Step 5: Create Retriever
# Search and retrieve information contained in the documents.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 1})

# Step 6: Create Prompt
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.

#Previous Chat History:
{chat_history}

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)

# Step 7: Load LLM
llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)

# Step 8: Create Chain
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

    Number of split chunks: 171


## 创建可管理对话历史的 chain


```python
# Dictionary to store session records
store = {}

# Function to retrieve session records based on session ID
def get_session_history(session_ids):
    print(f"[Conversation Session ID]: {session_ids}")
    if session_ids not in store:  # If the session ID is not in the store
        # Create a new ChatMessageHistory object and save it to the store
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  

# Create a RAG chain that records conversations
rag_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # Function to retrieve session history
    input_messages_key="question",  # Key for the template variable that will contain the user's question
    history_messages_key="chat_history",  # Key for the history messages
)
```

## 多轮 QA 测试


```python
rag_with_history.invoke(
    # Input question
    {"question": "What are the three key components necessary to achieve 'trustworthy AI' in the European approach to AI policy?"},
    # Record the conversation based on the session ID.
    config={"configurable": {"session_id": "rag123"}},
)
```

    [Conversation Session ID]: rag123





    "Based on the provided context, the three key components necessary to achieve 'trustworthy AI' in the European approach to AI policy are not explicitly stated. The context provided seems to be about improving public services and creating a secure, trusted data space, but it does not directly address the components of trustworthy AI. To provide an accurate answer, I would need more specific information from the document."




```python
rag_with_history.invoke(
    # Input question
    {"question": "Please translate the previous answer into Spanish."},
    # Record the conversation based on the session ID.
    config={"configurable": {"session_id": "rag123"}},
)
```

    [Conversation Session ID]: rag123





    'No se especifican los tres componentes clave necesarios para lograr la "IA confiable" en la aproximación europea a la política de IA en el contexto proporcionado. El contexto parece estar sobre mejorar los servicios públicos y crear un espacio de datos seguro y confiable, pero no aborda directamente los componentes de la IA confiable. Para proporcionar una respuesta precisa, necesitaría más información específica del documento.'



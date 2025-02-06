# **Document**  

用于存储一段文本及其相关元数据的类。  

- **`page_content`** （必需）：以字符串形式存储一段文本。  
- **`metadata`** （可选）：以字典形式存储与 `page_content` 相关的元数据。


```python
from langchain_core.documents import Document

document = Document(page_content="Hello, welcome to LangChain Open Tutorial!")

document
```




    Document(metadata={}, page_content='Hello, welcome to LangChain Open Tutorial!')



# **文档加载器（Document Loader）**  

**文档加载器**是一个用于从**各种来源**加载 `Document` 的类。  

以下是一些常见的文档加载器示例：  

- **`PyPDFLoader`** ：加载 PDF 文件  
- **`CSVLoader`** ：加载 CSV 文件  
- **`UnstructuredHTMLLoader`** ：加载 HTML 文件  
- **`JSONLoader`** ：加载 JSON 文件  
- **`TextLoader`** ：加载纯文本文件  
- **`DirectoryLoader`** ：从目录中批量加载文档  


```python
from langchain_community.document_loaders import PyPDFLoader

# Set up the loader
FILE_PATH = "./data/01-document-loader-sample.pdf"
loader = PyPDFLoader(FILE_PATH)
```

### **load()**  

- 加载文档，并以 `list[Document]` 的形式返回。


```python
docs = loader.load()
print(len(docs))
print('-'*3)
docs[0:2]
```

    48
    ---





    [Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 0}, page_content=' \n \n \nOctober 2016 \n \n \n \n \n \n \n \n \n \nTHE NATIONAL  \nARTIFICIAL INTELLIGENCE \nRESEARCH AND DEVELOPMENT \nSTRATEGIC PLAN  \nNational Science and Technology Council \n \nNetworking and Information Technology \nResearch and Development Subcommittee \n '),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 1}, page_content=' ii \n \n ')]



### **aload()**  

- **异步**加载文档，并以 `list[Document]` 的形式返回。


```python
# Load Documents asynchronously
docs = await loader.aload()
```

### **lazy_load()**  

- **顺序**加载文档，并以 `Iterator[Document]` 的形式返回。


```python
docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)
    break  # Used to limit the output length
```

### **alazy_load()**  

- **异步**顺序加载文档，并以 `AsyncIterator[Document]` 的形式返回。

可以观察到，这种方法作为一个 **`async_generator`** 工作。它是一种特殊类型的异步迭代器，能够**按需生成**值，而不需要一次性将所有值存储在内存中。


```python
loader.alazy_load()
docs = loader.alazy_load()
async for doc in docs:
    print(doc.metadata)
    break  # Used to limit the output length
```

### **load_and_split()**  

- 加载文档，并使用 `TextSplitter` **自动拆分**为多个文本块，最终以 `list[Document]` 的形式返回。


```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up the TextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=0)

# Split Documents into chunks
docs = loader.load_and_split(text_splitter=text_splitter)

print(len(docs))
docs[0:3]
```

    1430





    [Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 0}, page_content='October 2016 \n \n \n \n \n \n \n \n \n \nTHE NATIONAL  \nARTIFICIAL INTELLIGENCE \nRESEARCH AND DEVELOPMENT \nSTRATEGIC PLAN'),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 0}, page_content='National Science and Technology Council \n \nNetworking and Information Technology \nResearch and Development Subcommittee'),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 1}, page_content='ii')]



# PDF Loader

`PyPDF` 是最广泛使用的 Python 库之一，用于 PDF 处理。

在这里，我们使用 `pypdf` 将 PDF 加载为文档数组，每个文档包含一个 `page` 页码，并包含页面内容和元数据。

LangChain 的 [PyPDFLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFLoader.html) 集成了 PyPDF，将 PDF 文档解析为 LangChain 的 Document 对象。


```python
from langchain_community.document_loaders import PyPDFLoader

# Initialize the PDF loader
FILE_PATH = "./data/01-document-loader-sample.pdf"
loader = PyPDFLoader(FILE_PATH)

# Load data into Document objects
docs = loader.load()

# Print the contents of the document
print(docs[10].page_content[:300])
```

    NATIONAL ARTIFICIAL INTELLIGENCE RESEARCH AND DEVELOPMENT STRATEGIC PLAN 
     
     3 
    Executive Summary 
    Artificial intelligence (AI) is a transformative technology that holds promise for tremendous societal and 
    economic benefit. AI has the potential to revolutionize how we live, work, learn, discover, a


## PyPDF(OCR)

有些 PDF 包含扫描文档或图片中的文本图像。你也可以使用 `rapidocr-onnxruntime` 包从图像中提取文本。


```python
loader = PyPDFLoader(FILE_PATH, extract_images=True)
docs = loader.load()
```

## PyPDF Directory

从目录中导入所有 PDF 文档。


```python
from langchain_community.document_loaders import PyPDFDirectoryLoader

# directory path
loader = PyPDFDirectoryLoader("./data/")

# load documents
docs = loader.load()

# print the number of documents
print(len(docs))

```

    96


## **PyMuPDF**  

`PyMuPDF` 经过速度优化，并提供关于 PDF 及其页面的详细元数据。它**每页返回一个文档**。  

LangChain 的 [PyMuPDFLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyMuPDFLoader.html) 集成了 `PyMuPDF`，可将 PDF 文档解析为 LangChain 的 `Document` 对象。


```python
from langchain_community.document_loaders import PyMuPDFLoader

# create an instance of the PyMuPDF loader
FILE_PATH = "./data/01-document-loader-sample.pdf"
loader = PyMuPDFLoader(FILE_PATH)

# load the document
docs = loader.load()

# print the contents of the document
print(len(docs))
```

    48


    /data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/site-packages/langchain_community/document_loaders/parsers/pdf.py:322: UserWarning: Warning: Empty content on page 4 of document ./data/01-document-loader-sample.pdf
      warnings.warn(


# WebBaseLoader

`WebBaseLoader` 是 LangChain 中一个专门用于处理基于网页内容的文档加载器。

它利用 `BeautifulSoup4` 库有效地解析网页，并通过 `SoupStrainer` 和其他 `bs4` 参数提供可自定义的解析选项。


```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Load news article content using WebBaseLoader
loader = WebBaseLoader(
   web_paths=(
	"https://blog.csdn.net/wait_for_taht_day5/article/details/50570827",
	"https://blog.csdn.net/teethfairy/article/details/7287307"
	),
   encoding='utf-8'
)

# Load and process the documents
docs = loader.load()
print(f"Number of documents: {len(docs)}")

import re
print(re.sub(r'\n+', '\n', docs[0].page_content)[:100])
```

    Number of documents: 2
     
    【Python】Hello World 输入输出_python你好世界输入,输出英文-CSDN博客
    【Python】Hello World 输入输出
    最新推荐文章于 2024-12-04 08:5


# CSV Loader



```python
from langchain_community.document_loaders.csv_loader import CSVLoader

# Create CSVLoader instance
loader = CSVLoader(file_path="./data/titanic.csv")

# Load documents
docs = loader.load()

print(docs[0])
```

    page_content='PassengerId: 1
    Survived: 0
    Pclass: 3
    Name: Braund, Mr. Owen Harris
    Sex: male
    Age: 22
    SibSp: 1
    Parch: 0
    Ticket: A/5 21171
    Fare: 7.25
    Cabin: 
    Embarked: S' metadata={'source': './data/titanic.csv', 'row': 0}



```python
docs[0].page_content
```




    'PassengerId: 1\nSurvived: 0\nPclass: 3\nName: Braund, Mr. Owen Harris\nSex: male\nAge: 22\nSibSp: 1\nParch: 0\nTicket: A/5 21171\nFare: 7.25\nCabin: \nEmbarked: S'



它会读取 header 然后把每行数据重新组织

原始数据
```
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked 
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
```
---
```
PassengerId: 1\nSurvived: 0\nPclass: 3\nName: Braund, Mr. Owen Harris\nSex: male\nAge: 22\nSibSp: 1\nParch: 0\nTicket: A/5 21171\nFare: 7.25\nCabin: \nEmbarked: S
```


## **自定义 CSV 解析和加载**

`CSVLoader` 接受一个 `csv_args` 关键字参数，用于定制传递给 Python 的 `csv.DictReader` 的参数。这使得你可以处理各种 CSV 格式，如自定义分隔符、引号字符或特定的换行符处理。

有关支持的 `csv_args` 及如何根据你的特定需求定制解析的更多信息，请参见 Python 的 [csv 模块](https://docs.python.org/3/library/csv.html) 文档。


```python
loader = CSVLoader(
    file_path="./data/titanic.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": [
            "Passenger ID",
            "Survival (1: Survived, 0: Died)",
            "Passenger Class",
            "Name",
            "Sex",
            "Age",
            "Number of Siblings/Spouses Aboard",
            "Number of Parents/Children Aboard",
            "Ticket Number",
            "Fare",
            "Cabin",
            "Port of Embarkation",
        ],
    },
)

docs = loader.load()

print(docs[1].page_content)
```

    Passenger ID: 1
    Survival (1: Survived, 0: Died): 0
    Passenger Class: 3
    Name: Braund, Mr. Owen Harris
    Sex: male
    Age: 22
    Number of Siblings/Spouses Aboard: 1
    Number of Parents/Children Aboard: 0
    Ticket Number: A/5 21171
    Fare: 7.25
    Cabin: 
    Port of Embarkation: S



```python
loader = CSVLoader(
    file_path="./data/titanic.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": [
            "Passenger ID",
            "Survival (1: Survived, 0: Died)",
            "Passenger Class",
            "Name",
            "Sex",
            "Age",
        ],
    },
)

docs = loader.load()

print(docs[1].page_content)
```

    Passenger ID: 1
    Survival (1: Survived, 0: Died): 0
    Passenger Class: 3
    Name: Braund, Mr. Owen Harris
    Sex: male
    Age: 22
    None: 1,0,A/5 21171,7.25,,S


这些参数其实没啥好解释的, fieldnames 不是用于选择列的, 而是重新命名的, 如果部分列没命名, 那会分到 None 列

你应该使用 `source_column` 参数来指定每一行生成文档的来源。否则，`file_path` 将作为所有从 CSV 文件创建的文档的来源。

当在一个用于基于源信息回答问题的链中使用从 CSV 文件加载的文档时，这一点尤其有用。


```python
loader = CSVLoader(
    file_path="./data/titanic.csv",
    source_column="PassengerId",  # Specify the source column
)

docs = loader.load()  

print(docs[1])
```

    page_content='PassengerId: 2
    Survived: 1
    Pclass: 1
    Name: Cumings, Mrs. John Bradley (Florence Briggs Thayer)
    Sex: female
    Age: 38
    SibSp: 1
    Parch: 0
    Ticket: PC 17599
    Fare: 71.2833
    Cabin: C85
    Embarked: C' metadata={'source': '2', 'row': 1}


注意 metadata.source 变成了对应的 PassengerId

# **DataFrameLoader**  



`Pandas` 是一个开源的数据分析和处理工具，专为 Python 编程语言设计。该库在数据科学、机器学习以及多个领域中广泛应用，用于处理各种数据。

LangChain 的 `DataFrameLoader` 是一个强大的工具，旨在无缝地将 `Pandas` `DataFrame` 集成到 LangChain 工作流中。


```python
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd

df = pd.read_csv("./data/titanic.csv")
# df = pd.read_excel("./data/titanic.xlsx")

print(df.head(n=5))
# The Name column of the DataFrame is specified to be used as the content of each document.
loader = DataFrameLoader(df, page_content_column="Name")

docs = loader.load()

print(docs[0].page_content)

```

       PassengerId  Survived  Pclass  \
    0            1         0       3   
    1            2         1       1   
    2            3         1       3   
    3            4         1       1   
    4            5         0       3   
    
                                                    Name     Sex   Age  SibSp  \
    0                            Braund, Mr. Owen Harris    male  22.0      1   
    1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
    2                             Heikkinen, Miss. Laina  female  26.0      0   
    3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
    4                           Allen, Mr. William Henry    male  35.0      0   
    
       Parch            Ticket     Fare Cabin Embarked  
    0      0         A/5 21171   7.2500   NaN        S  
    1      0          PC 17599  71.2833   C85        C  
    2      0  STON/O2. 3101282   7.9250   NaN        S  
    3      0            113803  53.1000  C123        S  
    4      0            373450   8.0500   NaN        S  
    Braund, Mr. Owen Harris


可通过 page_content_column 来指定哪些 dataframe 的列被读取. csv 和 excel 文件都可以用 DataFrameLoader 加载

# **Docx2txtLoader**  


- 采用轻量级的 Python 模块 **`docx2txt`** 进行文本提取。  
- **快速**、**简单**地从 `.docx` 文件中提取文本。  
- 适用于**高效且直接**的文本处理任务。  


```python
from langchain_community.document_loaders import Docx2txtLoader

# Initialize the document loader
loader = Docx2txtLoader("data/sample-word-document_eng.docx")

# Load the document
docs = loader.load()

# Print the metadata of the document
print(f"Document Metadata: {docs[0].metadata}\n")

# Note: The entire docx file is converted into a single document.
# It needs to be split into smaller parts using a text splitter.
print('-'*5)
print(docs[0].page_content[:100])
```

    Document Metadata: {'source': 'data/sample-word-document_eng.docx'}
    
    -----
    Semantic Search
    
    
    
    Definition: Semantic search is a search methodology that goes beyond simple keywo


不像 PDF Loader 会默认根据页切分文档, Docx2txtLoader 会把一个文件读取为一个 Document

# TXT Loader


```python
from langchain_community.document_loaders import TextLoader

# Create a text loader
loader = TextLoader("data/appendix-keywords.txt", encoding="utf-8")

# Load the document
docs = loader.load()
print(f"Number of documents: {len(docs)}\n")
print("[Metadata]\n")
print(docs[0].metadata)
print("\n========= [Preview - First 500 Characters] =========\n")
print(docs[0].page_content[:500])
```

    Number of documents: 1
    
    [Metadata]
    
    {'source': 'data/appendix-keywords.txt'}
    
    ========= [Preview - First 500 Characters] =========
    
    Semantic Search
    
    Definition: Semantic search is a search method that goes beyond simple keyword matching by understanding the meaning of the user’s query to return relevant results.
    Example: If a user searches for “planets in the solar system,” the system might return information about related planets such as “Jupiter” or “Mars.”
    Related Keywords: Natural Language Processing, Search Algorithms, Data Mining
    
    Embedding
    
    Definition: Embedding is the process of converting textual data, such as words


# JSONLoader


```python
from langchain_community.document_loaders import JSONLoader

# Create JSONLoader
loader = JSONLoader(
    file_path="data/people.json",
    jq_schema=".people[]",  # Access each item in the people array
    text_content=False,
)

# Example: extract only contact_details
# loader = JSONLoader(
#     file_path="data/people.json",
#     jq_schema=".people[].contact_details",
#     text_content=False,
# )

# Or extract only hobbies from personal_preferences
# loader = JSONLoader(
#     file_path="data/people.json",
#     jq_schema=".people[].personal_preferences.hobbies",
#     text_content=False,
# )

# Load documents
docs = loader.load()
docs
```




    [Document(metadata={'source': '/data02/hyzhang10/pengxia2/LangChain-tutorial/data/people.json', 'seq_num': 1}, page_content='{"name": {"first": "Alice", "last": "Johnson"}, "age": 28, "contact": {"email": "alice.johnson@example.com", "phone": "+1-555-0123", "social_media": {"twitter": "@alice_j", "linkedin": "linkedin.com/in/alicejohnson"}}, "address": {"street": "123 Maple St", "city": "Springfield", "state": "IL", "zip": "62704", "country": "USA"}, "personal_preferences": {"hobbies": ["Reading", "Hiking", "Cooking"], "favorite_food": "Italian", "music_genre": "Jazz", "travel_destinations": ["Japan", "Italy", "Canada"]}, "interesting_fact": "Alice has traveled to over 15 countries and speaks 3 languages."}'),
     Document(metadata={'source': '/data02/hyzhang10/pengxia2/LangChain-tutorial/data/people.json', 'seq_num': 2}, page_content='{"name": {"first": "Bob", "last": "Smith"}, "age": 34, "contact": {"email": "bob.smith@example.com", "phone": "+1-555-0456", "social_media": {"twitter": "@bobsmith34", "linkedin": "linkedin.com/in/bobsmith"}}, "address": {"street": "456 Oak Ave", "city": "Metropolis", "state": "NY", "zip": "10001", "country": "USA"}, "personal_preferences": {"hobbies": ["Photography", "Cycling", "Video Games"], "favorite_food": "Mexican", "music_genre": "Rock", "travel_destinations": ["Brazil", "Australia", "Germany"]}, "interesting_fact": "Bob is an avid gamer and has competed in several national tournaments."}'),
     Document(metadata={'source': '/data02/hyzhang10/pengxia2/LangChain-tutorial/data/people.json', 'seq_num': 3}, page_content='{"name": {"first": "Charlie", "last": "Davis"}, "age": 45, "contact": {"email": "charlie.davis@example.com", "phone": "+1-555-0789", "social_media": {"twitter": "@charliedavis45", "linkedin": "linkedin.com/in/charliedavis"}}, "address": {"street": "789 Pine Rd", "city": "Gotham", "state": "NJ", "zip": "07001", "country": "USA"}, "personal_preferences": {"hobbies": ["Gardening", "Fishing", "Woodworking"], "favorite_food": "Barbecue", "music_genre": "Country", "travel_destinations": ["Canada", "New Zealand", "Norway"]}, "interesting_fact": "Charlie has a small farm where he raises chickens and grows organic vegetables."}'),
     Document(metadata={'source': '/data02/hyzhang10/pengxia2/LangChain-tutorial/data/people.json', 'seq_num': 4}, page_content='{"name": {"first": "Dana", "last": "Lee"}, "age": 22, "contact": {"email": "dana.lee@example.com", "phone": "+1-555-0111", "social_media": {"twitter": "@danalee22", "linkedin": "linkedin.com/in/danalee"}}, "address": {"street": "234 Birch Blvd", "city": "Star City", "state": "CA", "zip": "90001", "country": "USA"}, "personal_preferences": {"hobbies": ["Dancing", "Sketching", "Traveling"], "favorite_food": "Thai", "music_genre": "Pop", "travel_destinations": ["Thailand", "France", "Spain"]}, "interesting_fact": "Dana is a dance instructor and has won several local competitions."}'),
     Document(metadata={'source': '/data02/hyzhang10/pengxia2/LangChain-tutorial/data/people.json', 'seq_num': 5}, page_content='{"name": {"first": "Ethan", "last": "Garcia"}, "age": 31, "contact": {"email": "ethan.garcia@example.com", "phone": "+1-555-0999", "social_media": {"twitter": "@ethangarcia31", "linkedin": "linkedin.com/in/ethangarcia"}}, "address": {"street": "345 Cedar St", "city": "Central City", "state": "TX", "zip": "75001", "country": "USA"}, "personal_preferences": {"hobbies": ["Running", "Travel Blogging", "Cooking"], "favorite_food": "Indian", "music_genre": "Hip-Hop", "travel_destinations": ["India", "Italy", "Mexico"]}, "interesting_fact": "Ethan runs a popular travel blog where he shares his adventures and culinary experiences."}')]



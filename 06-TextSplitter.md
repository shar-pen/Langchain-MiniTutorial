# Character Text Splitter

`CharacterTextSplitter` 提供了高效的文本切分功能，带来了以下几个关键优势：

- **Token 限制：** 克服了 LLM 上下文窗口大小的限制。
- **搜索优化：** 实现更精确的基于块的检索。
- **内存效率：** 有效处理大规模文档。
- **上下文保持：** 通过 `chunk_overlap` 保持文本的连贯性。

`CharacterTextSplitter` 是通过将文本按指定的分隔符拆分成多个块来进行切分的。它的几个重要参数如下:

- **分隔符 (`separator`)**：首先，`CharacterTextSplitter` 会按照用户指定的分隔符（如换行符、空格、逗号等）拆分文本。分隔符可以是一个简单的字符串，也可以是一个正则表达式。
   
- **块大小 (`chunk_size`)**：每个生成的块会被限制在一个最大大小（`chunk_size`）内，超出这个大小的文本会被分割成新的块。

- **重叠部分 (`chunk_overlap`)**：为了保持上下文的连贯性，`CharacterTextSplitter` 可以在相邻块之间保持一定的字符重叠。重叠的字符数由 `chunk_overlap` 参数指定，通常用来避免分割导致的上下文丢失。

它的切分方式主要是基于字符级别的，具体步骤如下：
- 从文本的开头开始，按照指定的分隔符拆分文本。
- 每当拆分一个块时，会检查该块的字符数是否超过了 `chunk_size`。如果超过，则将其拆成更小的块，直到所有块的字符数都在 `chunk_size` 限制内。
- 若在块的边界有字符重叠（由 `chunk_overlap` 控制），则相邻块会共享这些重叠字符。


```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/appendix-keywords.txt", encoding="utf-8")
docs = loader.load()
```

创建 `CharacterTextSplitter` 并设置以下参数：

* `separator`: 用于拆分文本的字符串（例如，换行符、空格、自定义分隔符）
* `chunk_size`: 返回的每个块的最大大小
* `chunk_overlap`: 块与块之间的字符重叠部分
* `length_function`: 用于测量每个块长度的函数
* `is_separator_regex`: 布尔值，指示分隔符是否应作为正则表达式模式处理


```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
   separator=" ",           # Splits whenever a space is encountered in text
   chunk_size=250,          # Each chunk contains maximum 250 characters
   chunk_overlap=50,        # Two consecutive chunks share 50 characters
   length_function=len,     # Counts total characters in each chunk
   is_separator_regex=False # Uses space as literal separator, not as regex
)

```




    ['Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.']



1. 直接拆分 Document 实例


```python

doc_splits = text_splitter.split_documents(docs)

print(len(doc_splits))
```

2. 拆分 text 为 Document


```python
metadatas = [
   {"document": 1},
   {"document": 2},
]
texts = ['Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.'] * 2

documents = text_splitter.create_documents(
   texts=texts,  # List of texts to split
   metadatas=metadatas,  # Corresponding metadata
)
```

3. 只拆分 text


```python
text_splits = text_splitter.split_text('Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.')
```

# RecursiveCharacterTextSplitter

这是推荐的文本切分方式。

该方法通过接收一个字符列表作为参数来工作。
它尝试按照给定字符列表的顺序将文本拆分成更小的片段，直到这些片段变得非常小。
默认情况下，字符列表为 **['\n\n', '\n', ' ', ',']**。
它会递归地按照以下顺序进行拆分：**段落** -> **句子** -> **单词**。
这意味着段落（然后是句子，再然后是单词）被视为最具语义关联性的文本片段，因此我们尽量将它们保持在一起。

具体拆分方式：

1. **首先按段落切分**：它会尝试用 `\n\n`（段落分隔符）将文本分割成较小的块。如果文本块的长度超出了 `chunk_size` 限制，继续向下进入下一步。
   
2. **按换行符切分**：如果按段落切分后仍然没有满足 `chunk_size` 限制，文本将继续使用换行符 `\n` 进行进一步的切分。

3. **按其他分隔符切分**：如果仍然需要进一步缩小块的大小，它会根据设定的其他分隔符（如空格、逗号等）进行切分。

4. **按句子或单词切分**：如果文本块仍然过大，最后可能会按句子或单词进行切分，直到满足 `chunk_size` 限制。

通过这种递归方式，`RecursiveCharacterTextSplitter` 确保了文本被合理切分，同时尽可能保持语义上的一致性。

总结来说，这种拆分方式会优先保留语义上关联性较强的部分（如段落、句子和单词），以确保文本在拆分后仍然保持较强的上下文关联性。

使用 `RecursiveCharacterTextSplitter` 将文本拆分为小块的示例：

- 将 **chunk_size** 设置为 250，以限制每个块的大小。
- 将 **chunk_overlap** 设置为 50，以允许相邻块之间有 50 个字符的重叠。
- 使用 **len** 函数作为 **length_function** 来计算文本的长度。
- 将 **is_separator_regex** 设置为 **False**，以禁用正则表达式作为分隔符的使用。


```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/appendix-keywords.txt", encoding="utf-8")
docs = loader.load()
```


```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set the chunk size to very small. These settings are for illustrative purposes only.
    chunk_size=250,
    # Sets the number of overlapping characters between chunks.
    chunk_overlap=50,
    # Specifies a function to calculate the length of the string.
    length_function=len,
    # Sets whether to use regular expressions as delimiters.
    is_separator_regex=False,
)

```

切分方式还是有类似的三种


```python
# 1.
doc_splits = text_splitter.split_documents(docs)

# 2.
doc_splits = text_splitter.create_documents(
	texts=['*'*20]*2,
	metadatas=[{"document": i} for i in range(2)]
)

# 3.
text_splits = text_splitter.split_text(docs[0].page_content)
```

# Text Splitting Methods in NLP


1. **基于词元的切分**  
   - **Tiktoken**：OpenAI的高性能BPE（字节对编码）分词器  
   - **Hugging Face tokenizers**：针对各种预训练模型的分词器  

2. **基于句子的切分**  
   - **SentenceTransformers**：在保持语义一致性的情况下对文本进行切分  
   - **NLTK**：基于自然语言处理的句子和单词切分  
   - **spaCy**：利用先进的语言处理能力进行文本切分  

3. **语言特定的工具**  
   - **KoNLPy**：用于韩文文本处理的专业切分工具  

每个工具都有其独特的特点和优势：  
- **Tiktoken** 提供快速的处理速度，并与OpenAI模型兼容  
- **SentenceTransformers** 提供基于语义的句子切分  
- **NLTK** 和 **spaCy** 实现了基于语言学规则的切分  
- **KoNLPy** 专注于韩文的形态学分析和切分

## tiktoken 

`tiktoken` 是OpenAI创建的一个快速BPE（字节对编码）分词器。


```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/appendix-keywords.txt", encoding="utf-8")
docs = loader.load()
```


```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    # Set the chunk size to 300.
    chunk_size=300,
    # Ensure there is no overlap between chunks.
    chunk_overlap=50,
)
# Split the file text into chunks.
doc_splits = text_splitter.split_documents(docs)
```

`CharacterTextSplitter.from_tiktoken_encoder` 和 `CharacterTextSplitter` 的主要区别在于它们如何计算文本的“长度”或“单位”：

1. **`CharacterTextSplitter`**:
   - 计算的是字符数量。它按字符的数量来切分文本。
   - 使用 `chunk_size` 和 `chunk_overlap` 参数时，切分的依据是字符数，而不是 token 数量。
   - 适用于不涉及复杂文本编码的场景，尤其是处理纯文本时。

2. **`CharacterTextSplitter.from_tiktoken_encoder`**:
   - 计算的是 token 数量，而不是字符数量。
   - 这个方法使用了 `tiktoken` 编码器来计算文本中的 token 数量，因此它可以准确处理各种文本编码方式，尤其是在与语言模型（如OpenAI的GPT模型）交互时，token数量更为关键。
   - 适用于需要考虑 token 数量限制的场景，特别是当你要确保文本能够适配特定模型的输入大小时。

简单来说，`CharacterTextSplitter` 按字符进行切分，而 `CharacterTextSplitter.from_tiktoken_encoder` 按 token 进行切分，后者更适合用于与大型语言模型交互时的文本预处理。

- 使用 `CharacterTextSplitter.from_tiktoken_encoder` 时，文本仅由 `CharacterTextSplitter` 进行切分，`Tiktoken` 分词器仅用于测量和合并切分后的文本。（这意味着切分后的文本可能会超过 `Tiktoken` 分词器测量的 chunk 大小。）
- 使用 `RecursiveCharacterTextSplitter.from_tiktoken_encoder` 时，确保切分后的文本不会超过语言模型允许的 chunk 大小。如果切分后的文本超过此大小，则会进行递归切分。此外，您还可以直接加载 `Tiktoken` 分词器，它会确保每个切分后的文本都小于 chunk 大小。

注意他们都不是直接基于 token 来切分的, 只用于测量文本长度

## `TokenTextSplitter`

使用 `TokenTextSplitter` 类将文本分割成基于 token 的块。


```python
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/appendix-keywords.txt", encoding="utf-8")
docs = loader.load()

text_splitter = TokenTextSplitter(
    chunk_size=200,  # Set the chunk size to 10.
    chunk_overlap=50,  # Set the overlap between chunks to 0.
)

# Split the state_of_the_union text into chunks.
doc_splits = text_splitter.split_documents(docs)
print(doc_splits[0].page_content)  # Print the first chunk of the divided text.
```

    Semantic Search
    
    Definition: Semantic search is a search method that goes beyond simple keyword matching by understanding the meaning of the user’s query to return relevant results.
    Example: If a user searches for “planets in the solar system,” the system might return information about related planets such as “Jupiter” or “Mars.”
    Related Keywords: Natural Language Processing, Search Algorithms, Data Mining
    
    Embedding
    
    Definition: Embedding is the process of converting textual data, such as words or sentences, into low-dimensional continuous vectors. This allows computers to better understand and process the text.
    Example: The word “apple” might be represented as a vector like [0.65, -0.23, 0.17].
    Related Keywords: Natural Language Processing, Vectorization, Deep Learning
    
    Token
    
    Definition: A token refers to a smaller unit of text obtained


## spaCy

spaCy 是一个用于高级自然语言处理的开源软件库，使用 Python 和 Cython 编程语言编写。

除了 NLTK，另一个替代方案是使用 spaCy 分词器。

1. 文本如何被分割：文本使用 spaCy 分词器进行分割。
2. 块的大小如何测量：它通过字符数来测量。

执行以下代码前需提前下载 SpaCy en_core_web_sm 模型

`python -m spacy download en_core_web_sm --quiet`


```python
import warnings
from langchain_text_splitters import SpacyTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/appendix-keywords.txt", encoding="utf-8")
docs = loader.load()

# Ignore  warning messages.
warnings.filterwarnings("ignore")

# Create the SpacyTextSplitter.
text_splitter = SpacyTextSplitter(
    chunk_size=200,  # Set the chunk size to 200.
    chunk_overlap=50,  # Set the overlap between chunks to 50.
)

doc_splits = text_splitter.split_documents(docs)
print(doc_splits[0].page_content)
```

## SentenceTransformers

`SentenceTransformersTokenTextSplitter` 是一个专门为 `sentence-transformer` 模型设计的文本切分器。

它的默认行为是将文本分割成适合当前使用的 `sentence-transformer` 模型的 token 窗口大小的块。


```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/appendix-keywords.txt", encoding="utf-8")
docs = loader.load()

# Create a sentence splitter and set the overlap between chunks to 50.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
splitter = SentenceTransformersTokenTextSplitter(
	chunk_size=200, 
	chunk_overlap=50, 
	model_name='../DataCollection/officials/bge-large-zh-v1.5/'
	)
doc_splits = splitter.split_documents(docs)
print(doc_splits[0].page_content)
```


```python
# the number of start and stop tokens is 2.
text_token_count = splitter.count_tokens(text=doc_splits[0].page_content) - 2
print(text_token_count)
```

## HuggingFace Tokenizer

HuggingFace 提供了各种分词器。

这段代码展示了如何使用 HuggingFace 的 `AutoTokenizer` 分词器来计算文本的 token 长度。


```python
from transformers import AutoTokenizer

hf_tokenizer = AutoTokenizer.from_pretrained('../DataCollection/officials/Qwen2.5-14B-Instruct')
```


```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("data/appendix-keywords.txt", encoding="utf-8")
docs = loader.load()


text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
	tokenizer=hf_tokenizer,
    chunk_size=300,
    chunk_overlap=50,
)

doc_splits = text_splitter.split_documents(docs)
print(doc_splits[0].page_content)
```

    Semantic Search
    
    Definition: Semantic search is a search method that goes beyond simple keyword matching by understanding the meaning of the user’s query to return relevant results.
    Example: If a user searches for “planets in the solar system,” the system might return information about related planets such as “Jupiter” or “Mars.”
    Related Keywords: Natural Language Processing, Search Algorithms, Data Mining
    
    Embedding
    
    Definition: Embedding is the process of converting textual data, such as words or sentences, into low-dimensional continuous vectors. This allows computers to better understand and process the text.
    Example: The word “apple” might be represented as a vector like [0.65, -0.23, 0.17].
    Related Keywords: Natural Language Processing, Vectorization, Deep Learning
    
    Token
    
    Definition: A token refers to a smaller unit of text obtained by splitting a larger text. It can be a word, sentence, or phrase.
    Example: The sentence “I go to school” can be split into tokens: “I”, “go”, “to”, “school”.
    Related Keywords: Tokenization, Natural Language Processing, Parsing
    
    Tokenizer
    
    Definition: A tokenizer is a tool that splits text data into tokens. It is commonly used in natural language processing for data preprocessing.
    Example: The sentence “I love programming.” can be tokenized into [“I”, “love”, “programming”, “.”].
    Related Keywords: Tokenization, Natural Language Processing, Parsing


# Prompt Template

Prompt 模板对于生成动态且灵活的提示至关重要，可用于各种场景，例如会话历史记录、结构化输出和特定查询。  

在本教程中，我们将探讨创建 `PromptTemplate` 对象的方法，应用部分变量，通过 YAML 文件管理模板，并利用 `ChatPromptTemplate` 和 `MessagePlaceholder` 等高级工具来增强功能。


```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)
```

## 创建 `PromptTemplate` 对象  

有两种方法可以创建 `PromptTemplate` 对象：  
- **1.** 使用 `from_template()` 方法。  
- **2.** 直接创建 `PromptTemplate` 对象并同时生成提示词。

### **方法 1. 使用 `from_template()` 方法**  

- 使用 `{variable}` 语法定义模板，其中 `variable` 代表可替换的变量。


```python
from langchain_core.prompts import PromptTemplate

# {}内部是变量
template = "What is the capital of {country}?"

# 使用`from_template`函数来创建模板
prompt = PromptTemplate.from_template(template)
prompt
```




    PromptTemplate(input_variables=['country'], input_types={}, partial_variables={}, template='What is the capital of {country}?')



```
PromptTemplate(input_variables=['country'], input_types={}, partial_variables={}, template='What is the capital of {country}?')
```
类已经解析出country这个变量，可以通过为变量 `country` 赋值来完成提示词。


```python
# 类似str的`format`方法来创建实例
prompt.format(country="United States of America")
```




    'What is the capital of United States of America?'



进一步用chain来简化流程


```python
template = "What is the capital of {country}?"
prompt = PromptTemplate.from_template(template)
chain = prompt | llm
chain.invoke("United States of America").content
```




    'The capital of the United States of America is Washington, D.C.'



### **方法 2. 直接创建 `PromptTemplate` 对象并同时生成提示**  

- **明确指定 `input_variables`** 以进行额外的验证。  
- 否则，如果 `input_variables` 与模板字符串中的变量不匹配，实例化时可能会引发异常。


```python
from langchain_core.prompts import PromptTemplate
# Define template
template = "What is the capital of {country}?"

# Create a prompt template with `PromptTemplate` object
prompt = PromptTemplate(
    template=template,
    input_variables=["country"],
)
prompt
```




    PromptTemplate(input_variables=['country'], input_types={}, partial_variables={}, template='What is the capital of {country}?')



### partial variables

可临时固定的可变参数, 是特殊的 `input_variables`, 是对应 `input_variables` 在缺失时的默认值。
使用 `partial_variables`，您可以**部分应用函数**。这在需要共享 **通用变量** 时特别有用。  

**常见示例：**  
- **日期或时间（date or time）** 是典型的应用场景。  

例如，假设您希望在提示中指定当前日期：  
- **直接硬编码日期** 或 **每次手动传递日期变量** 可能不太灵活。  
- **更好的方法** 是使用一个返回当前日期的函数，将其部分应用于提示模板，从而动态填充日期变量，使提示更具适应性。


```python
from langchain_core.prompts import PromptTemplate
# Define template
template = "What are the capitals of {country1} and {country2}, respectively?"

# Create a prompt template with `PromptTemplate` object
prompt = PromptTemplate(
    template=template,
    input_variables=["country1"],
    partial_variables={
        "country2": "United States of America"  # Pass `partial_variables` in dictionary form
    },
)
prompt
```




    PromptTemplate(input_variables=['country1'], input_types={}, partial_variables={'country2': 'United States of America'}, template='What are the capitals of {country1} and {country2}, respectively?')




```python
prompt.format(country1="South Korea")
```




    'What are the capitals of South Korea and United States of America, respectively?'



通过`partial()`函数修改或者增加临时变量, 或者直接修改 PromptTemplate.partial_variables
- prompt_partial = prompt.partial(country2="India"), 可创建新实例的同时保留原实例
- prompt.partial_variables = {'country2':'china'}, 直接修改原实例


```python
prompt_partial = prompt.partial(country2="India")
prompt_partial.format(country1="South Korea")
```




    'What are the capitals of South Korea and India, respectively?'




```python
prompt.partial_variables = {'country2':'china'}
prompt.format(country1="South Korea")
```




    'What are the capitals of South Korea and china, respectively?'



`partial variables` 可以临时用新值, 不会影响缺失时的默认值


```python
print(prompt_partial.format(country1="South Korea", country2="Canada"))
print(prompt_partial.format(country1="South Korea"))
```

    What are the capitals of South Korea and Canada, respectively?
    What are the capitals of South Korea and India, respectively?


`partial variables` 可用函数传递, 不需要手动设置新值


```python
from datetime import datetime

def get_today():
    return datetime.now().strftime("%B %d")

prompt = PromptTemplate(
    template="Today's date is {today}. Please list {n} celebrities whose birthday is today. Please specify their date of birth.",
    input_variables=["n"],
    partial_variables={
        "today": get_today  # Pass `partial_variables` in dictionary form
    },
)

prompt.format(n=3)
```




    "Today's date is January 30. Please list 3 celebrities whose birthday is today. Please specify their date of birth."



## **从 YAML 文件加载 Prompt 模板**  

您可以将 **Prompt 模板** 存储在单独的 **YAML 文件** 中，并使用 `load_prompt` 进行加载和管理。

以下是一个yaml示例: 

---

```yaml
_type: "prompt"
template: "What is the color of {fruit}?"
input_variables: ["fruit"]
```
---


```python
from langchain_core.prompts import load_prompt

prompt = load_prompt("prompts/fruit_color.yaml", encoding="utf-8")
prompt
```

## **ChatPromptTemplate**  

`ChatPromptTemplate` 可用于将**会话历史记录**包含到提示词中，以提供上下文信息。消息以 **(`role`, `message`)** 元组的形式组织，并存储在 **列表** 中。

**角色（role）**:
- **`"system"`** ：系统设置信息，通常用于全局指令或设定 AI 的行为。  
- **`"human"`** ：用户输入的消息。  
- **`"ai"`** ：AI 生成的响应消息。  


```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")
chat_prompt
```




    ChatPromptTemplate(input_variables=['country'], input_types={}, partial_variables={}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['country'], input_types={}, partial_variables={}, template='What is the capital of {country}?'), additional_kwargs={})])



ChatPromptTemplate(input_variables=['country'], input_types={}, partial_variables={}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['country'], input_types={}, partial_variables={}, template='What is the capital of {country}?'), additional_kwargs={})])

注意这个prompt被 `HumanMessagePromptTemplate`包装了，而且位于一个list中


```python
chat_prompt.format(country="United States of America")
```




    'Human: What is the capital of United States of America?'



### 多角色

使用 `ChatPromptTemplate.from_messages`来定义模板, 内部是 chat list, 每个 chat 都是以 **(`role`, `message`)** 元组的形式组织


```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        # role, message
        ("system", "You are a friendly AI assistant. Your name is {name}."),
        ("human", "Nice to meet you!"),
        ("ai", "Hello! How can I assist you?"),
        ("human", "{user_input}"),
    ]
)

# Create chat messages
messages = chat_template.format_messages(name="Teddy", user_input="What is your name?")
messages
```




    [SystemMessage(content='You are a friendly AI assistant. Your name is Teddy.', additional_kwargs={}, response_metadata={}),
     HumanMessage(content='Nice to meet you!', additional_kwargs={}, response_metadata={}),
     AIMessage(content='Hello! How can I assist you?', additional_kwargs={}, response_metadata={}),
     HumanMessage(content='What is your name?', additional_kwargs={}, response_metadata={})]



可直接用上面的 Message list 的形式调用大模型


```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)
llm.invoke(messages).content
```




    "My name is Teddy. It's nice to meet you! How can I help you today?"



### **MessagePlaceholder**  

`LangChain` 提供了 **`MessagePlaceholder`**，用途包括:
- **当不确定使用哪些角色** 作为消息提示模板的一部分时，它可以提供灵活性。  
- **在格式化时插入一组消息列表**，适用于动态会话历史记录的场景。


```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a summarization specialist AI assistant. Your mission is to summarize conversations using key points.",
        ),
        MessagesPlaceholder(variable_name="conversation"),
        ("human", "Summarize the conversation so far in {word_count} words."),
    ]
)
chat_prompt
```




    ChatPromptTemplate(input_variables=['conversation', 'word_count'], input_types={'conversation': list[typing.Annotated[typing.Union[typing.Annotated[langchain_core.messages.ai.AIMessage, Tag(tag='ai')], typing.Annotated[langchain_core.messages.human.HumanMessage, Tag(tag='human')], typing.Annotated[langchain_core.messages.chat.ChatMessage, Tag(tag='chat')], typing.Annotated[langchain_core.messages.system.SystemMessage, Tag(tag='system')], typing.Annotated[langchain_core.messages.function.FunctionMessage, Tag(tag='function')], typing.Annotated[langchain_core.messages.tool.ToolMessage, Tag(tag='tool')], typing.Annotated[langchain_core.messages.ai.AIMessageChunk, Tag(tag='AIMessageChunk')], typing.Annotated[langchain_core.messages.human.HumanMessageChunk, Tag(tag='HumanMessageChunk')], typing.Annotated[langchain_core.messages.chat.ChatMessageChunk, Tag(tag='ChatMessageChunk')], typing.Annotated[langchain_core.messages.system.SystemMessageChunk, Tag(tag='SystemMessageChunk')], typing.Annotated[langchain_core.messages.function.FunctionMessageChunk, Tag(tag='FunctionMessageChunk')], typing.Annotated[langchain_core.messages.tool.ToolMessageChunk, Tag(tag='ToolMessageChunk')]], FieldInfo(annotation=NoneType, required=True, discriminator=Discriminator(discriminator=<function _get_type at 0x7ff1a966cfe0>, custom_error_type=None, custom_error_message=None, custom_error_context=None))]]}, partial_variables={}, messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template='You are a summarization specialist AI assistant. Your mission is to summarize conversations using key points.'), additional_kwargs={}), MessagesPlaceholder(variable_name='conversation'), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['word_count'], input_types={}, partial_variables={}, template='Summarize the conversation so far in {word_count} words.'), additional_kwargs={})])




```python
formatted_chat_prompt = chat_prompt.format(
    word_count=5,
    conversation=[
        ("human", "Hello! I’m Teddy. Nice to meet you."),
        ("ai", "Nice to meet you! I look forward to working with you."),
    ],
)

print(formatted_chat_prompt)
```

    System: You are a summarization specialist AI assistant. Your mission is to summarize conversations using key points.
    Human: Hello! I’m Teddy. Nice to meet you.
    AI: Nice to meet you! I look forward to working with you.
    Human: Summarize the conversation so far in 5 words.


## **Few-Shot Prompting**  

LangChain 的 **Few-Shot Prompting** 提供了一种强大的框架，通过提供精心挑选的示例，引导语言模型生成高质量的输出。此技术**减少了大量模型微调的需求**，同时确保在各种应用场景中提供**精准且符合上下文**的结果。  

- **Few-Shot Prompt 模板**：  
  - 通过嵌入示例定义提示的结构和格式，指导模型生成一致的输出。  

- **示例选择策略（Example Selection Strategies）**：  
  - **动态选择最相关的示例** 以匹配特定查询，增强模型的上下文理解能力，提高响应准确性。  

- **Chroma 向量存储（Chroma Vector Store）**：  
  - 用于存储和检索基于**语义相似度**的示例，提供**可扩展且高效**的 Prompt 结构构建。

### **FewShotPromptTemplate**  

**Few-shot prompting** 是一种强大的技术，它通过提供**少量精心设计的示例**，引导语言模型生成**准确且符合上下文**的输出。LangChain 的 **`FewShotPromptTemplate`** 简化了这一过程，使用户能够**构建灵活且可复用的提示**，适用于问答、摘要、文本校正等任务。  

**1. 设计 Few-Shot 提示（Designing Few-Shot Prompts）**  
- **定义示例**，展示所需的输出结构和风格。  
- **确保示例覆盖边界情况**，以增强模型的理解能力和性能。  

**2. 动态示例选择（Dynamic Example Selection）**  
- **利用语义相似性或向量搜索**，选择最相关的示例，以匹配输入查询。  

**3. 集成 Few-Shot 提示（Integrating Few-Shot Prompts）**  
- **结合 Prompt 模板与语言模型**，构建强大的链式调用，以生成高质量的响应。  


```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)

# User query
question = "What is the capital of United States of America?"

# Query the model
response = llm.invoke(question)

# Print the response
print(response.content)
```

    The capital of the United States of America is Washington, D.C.


以下是一个 CoT 的示例prompt


```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# Define examples for the few-shot prompt
examples = [
    {
        "question": "Who lived longer, Steve Jobs or Einstein?",
        "answer": """Does this question require additional questions: Yes.
Additional Question: At what age did Steve Jobs die?
Intermediate Answer: Steve Jobs died at the age of 56.
Additional Question: At what age did Einstein die?
Intermediate Answer: Einstein died at the age of 76.
The final answer is: Einstein
""",
    },
    {
        "question": "When was the founder of Naver born?",
        "answer": """Does this question require additional questions: Yes.
Additional Question: Who is the founder of Naver?
Intermediate Answer: Naver was founded by Lee Hae-jin.
Additional Question: When was Lee Hae-jin born?
Intermediate Answer: Lee Hae-jin was born on June 22, 1967.
The final answer is: June 22, 1967
""",
    },
    {
        "question": "Who was the reigning king when Yulgok Yi's mother was born?",
        "answer": """Does this question require additional questions: Yes.
Additional Question: Who is Yulgok Yi's mother?
Intermediate Answer: Yulgok Yi's mother is Shin Saimdang.
Additional Question: When was Shin Saimdang born?
Intermediate Answer: Shin Saimdang was born in 1504.
Additional Question: Who was the king of Joseon in 1504?
Intermediate Answer: The king of Joseon in 1504 was Yeonsangun.
The final answer is: Yeonsangun
""",
    },
    {
        "question": "Are the directors of Oldboy and Parasite from the same country?",
        "answer": """Does this question require additional questions: Yes.
Additional Question: Who is the director of Oldboy?
Intermediate Answer: The director of Oldboy is Park Chan-wook.
Additional Question: Which country is Park Chan-wook from?
Intermediate Answer: Park Chan-wook is from South Korea.
Additional Question: Who is the director of Parasite?
Intermediate Answer: The director of Parasite is Bong Joon-ho.
Additional Question: Which country is Bong Joon-ho from?
Intermediate Answer: Bong Joon-ho is from South Korea.
The final answer is: Yes
""",
    },
]

example_prompt = PromptTemplate.from_template(
    "Question:\n{question}\nAnswer:\n{answer}"
)

# Print the first formatted example
print(example_prompt.format(**examples[0]))
```

    Question:
    Who lived longer, Steve Jobs or Einstein?
    Answer:
    Does this question require additional questions: Yes.
    Additional Question: At what age did Steve Jobs die?
    Intermediate Answer: Steve Jobs died at the age of 56.
    Additional Question: At what age did Einstein die?
    Intermediate Answer: Einstein died at the age of 76.
    The final answer is: Einstein
    


以下这个 `FewShotPromptTemplate` 将 examples 以 example_prompt 格式添加到真正 QA 的前面。真正的 QA 按照 suffix 格式展示


```python
# Initialize the FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question:\n{question}\nAnswer:",
    input_variables=["question"],
)

# Example question
question = "How old was Bill Gates when Google was founded?"

# Generate the final prompt
final_prompt = few_shot_prompt.format(question=question)
print(final_prompt)
```

    Question:
    Who lived longer, Steve Jobs or Einstein?
    Answer:
    Does this question require additional questions: Yes.
    Additional Question: At what age did Steve Jobs die?
    Intermediate Answer: Steve Jobs died at the age of 56.
    Additional Question: At what age did Einstein die?
    Intermediate Answer: Einstein died at the age of 76.
    The final answer is: Einstein
    
    
    Question:
    When was the founder of Naver born?
    Answer:
    Does this question require additional questions: Yes.
    Additional Question: Who is the founder of Naver?
    Intermediate Answer: Naver was founded by Lee Hae-jin.
    Additional Question: When was Lee Hae-jin born?
    Intermediate Answer: Lee Hae-jin was born on June 22, 1967.
    The final answer is: June 22, 1967
    
    
    Question:
    Who was the reigning king when Yulgok Yi's mother was born?
    Answer:
    Does this question require additional questions: Yes.
    Additional Question: Who is Yulgok Yi's mother?
    Intermediate Answer: Yulgok Yi's mother is Shin Saimdang.
    Additional Question: When was Shin Saimdang born?
    Intermediate Answer: Shin Saimdang was born in 1504.
    Additional Question: Who was the king of Joseon in 1504?
    Intermediate Answer: The king of Joseon in 1504 was Yeonsangun.
    The final answer is: Yeonsangun
    
    
    Question:
    Are the directors of Oldboy and Parasite from the same country?
    Answer:
    Does this question require additional questions: Yes.
    Additional Question: Who is the director of Oldboy?
    Intermediate Answer: The director of Oldboy is Park Chan-wook.
    Additional Question: Which country is Park Chan-wook from?
    Intermediate Answer: Park Chan-wook is from South Korea.
    Additional Question: Who is the director of Parasite?
    Intermediate Answer: The director of Parasite is Bong Joon-ho.
    Additional Question: Which country is Bong Joon-ho from?
    Intermediate Answer: Bong Joon-ho is from South Korea.
    The final answer is: Yes
    
    
    Question:
    How old was Bill Gates when Google was founded?
    Answer:



```python
response = llm.invoke(final_prompt)
print(response.content)
```

    Does this question require additional questions: Yes.
    Additional Question: When was Google founded?
    Intermediate Answer: Google was founded in 1998.
    Additional Question: When was Bill Gates born?
    Intermediate Answer: Bill Gates was born on October 28, 1955.
    The final answer is: Bill Gates was 43 years old when Google was founded.


## 特殊 prompt

### **RAG 文档分析**  

基于检索到的文档上下文**处理并回答问题**，确保**高准确性和高相关性**。


```python
from langchain.prompts import ChatPromptTemplate


system = """You are a precise and helpful AI assistant specializing in question-answering tasks based on provided context.
Your primary task is to:
1. Analyze the provided context thoroughly
2. Answer questions using ONLY the information from the context
3. Preserve technical terms and proper nouns exactly as they appear
4. If the answer cannot be found in the context, respond with: 'The provided context does not contain information to answer this question.'
5. Format responses in clear, readable paragraphs with relevant examples when available
6. Focus on accuracy and clarity in your responses
"""

human = """#Question:
{question}

#Context:
{context}

#Answer:
Please provide a focused, accurate response that directly addresses the question using only the information from the provided context."""

prompt = ChatPromptTemplate.from_messages(
	[
		("system", system), 
		("human", human)
	]
)

prompt
```




    ChatPromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template="You are a precise and helpful AI assistant specializing in question-answering tasks based on provided context.\nYour primary task is to:\n1. Analyze the provided context thoroughly\n2. Answer questions using ONLY the information from the context\n3. Preserve technical terms and proper nouns exactly as they appear\n4. If the answer cannot be found in the context, respond with: 'The provided context does not contain information to answer this question.'\n5. Format responses in clear, readable paragraphs with relevant examples when available\n6. Focus on accuracy and clarity in your responses\n"), additional_kwargs={}), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template='#Question:\n{question}\n\n#Context:\n{context}\n\n#Answer:\nPlease provide a focused, accurate response that directly addresses the question using only the information from the provided context.'), additional_kwargs={})])



### **具有来源归因的 RAG（RAG with Source Attribution）**  

增强型 RAG 实现，支持**详细的来源追踪和引用**，以提高**可追溯性和验证可靠性**。


```python
from langchain.prompts import ChatPromptTemplate


system = """You are a precise and thorough AI assistant that provides well-documented answers with source attribution.
Your responsibilities include:
1. Analyzing provided context thoroughly
2. Generating accurate answers based solely on the given context
3. Including specific source references for each key point
4. Preserving technical terminology exactly as presented
5. Maintaining clear citation format [source: page/document]
6. If information is not found in the context, state: 'The provided context does not contain information to answer this question.'

Format your response as:
1. Main Answer
2. Sources Used (with specific locations)
3. Confidence Level (High/Medium/Low)"""

human = """#Question:
{question}

#Context:
{context}

#Answer:
Please provide a detailed response with source citations using only information from the provided context."""

prompt = ChatPromptTemplate.from_messages(
	[
		("system", system), 
		("human", human)
	]
)
PROMPT_OWNER = "eun"
hub.push(f"{PROMPT_OWNER}/{prompt_title}", prompt, new_repo_is_public=True)
```

其实在回答要求里加入了源引用的要求

### **LLM 响应评估（LLM Response Evaluation）**  

基于**多项质量指标**对 LLM 响应进行**全面评估**，并提供**详细的评分方法**。


```python
from langchain.prompts import PromptTemplate


evaluation_prompt = """Evaluate the LLM's response based on the following criteria:

INPUT:
Question: {question}
Context: {context}
LLM Response: {answer}

EVALUATION CRITERIA:
1. Accuracy (0-10)
- Perfect (10): Completely accurate, perfectly aligned with context
- Good (7-9): Minor inaccuracies
- Fair (4-6): Some significant inaccuracies
- Poor (0-3): Major inaccuracies or misalignment

2. Completeness (0-10)
- Perfect (10): Comprehensive coverage of all relevant points
- Good (7-9): Covers most important points
- Fair (4-6): Missing several key points
- Poor (0-3): Critically incomplete

3. Context Relevance (0-10)
- Perfect (10): Optimal use of context
- Good (7-9): Good use with minor omissions
- Fair (4-6): Partial use of relevant context
- Poor (0-3): Poor context utilization

4. Clarity (0-10)
- Perfect (10): Exceptionally clear and well-structured
- Good (7-9): Clear with minor issues
- Fair (4-6): Somewhat unclear
- Poor (0-3): Confusing or poorly structured

SCORING METHOD:
1. Calculate individual scores
2. Compute weighted average:
   - Accuracy: 40%
   - Completeness: 25%
   - Context Relevance: 25%
   - Clarity: 10%
3. Normalize to 0-1 scale

OUTPUT FORMAT:
{
    "individual_scores": {
        "accuracy": float,
        "completeness": float,
        "context_relevance": float,
        "clarity": float
    },
    "weighted_score": float,
    "normalized_score": float,
    "evaluation_notes": string
}

Return ONLY the normalized_score as a decimal between 0 and 1."""

prompt = PromptTemplate.from_template(evaluation_prompt)
```

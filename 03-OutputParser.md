# PydanticOutputParser



`PydanticOutputParser` 是一个用于**将语言模型的输出转换为结构化信息**的类。它能够**提供清晰且有组织的格式化信息**，而不仅仅是简单的文本响应。  

通过使用此类，您可以**将语言模型的输出转换为特定的数据模型**，使其更易于处理和利用。  

---

## **主要方法**  

`PydanticOutputParser` 主要依赖于**两个核心方法**：  

**1. `get_format_instructions()`**  
- 提供指令，定义语言模型应输出的数据格式。  
- 例如，可以返回一个字符串，其中描述了数据字段及其格式要求。  
- 这些指令对于**让语言模型结构化输出**并**符合特定数据模型**至关重要。  

**2. `parse()`**  
- 接收语言模型的输出（通常是字符串），并将其**解析和转换**为特定的数据结构。  
- 使用 **Pydantic** 进行数据验证，将输入字符串与预定义的模式匹配，并转换为符合该模式的数据结构。


```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)
```

以下是一个使用 Parser 简化流程的示例


```python
email_conversation = """
From: John (John@bikecorporation.me)
To: Kim (Kim@teddyinternational.me)
Subject: “ZENESIS” bike distribution cooperation and meeting schedule proposal
Dear Mr. Kim,

I am John, Senior Executive Director at Bike Corporation. I recently learned about your new bicycle model, "ZENESIS," through your press release. Bike Corporation is a company that leads innovation and quality in the field of bicycle manufacturing and distribution, with long-time experience and expertise in this field.

We would like to request a detailed brochure for the ZENESIS model. In particular, we need information on technical specifications, battery performance, and design aspects. This information will help us further refine our proposed distribution strategy and marketing plan.

Additionally, to discuss the possibilities for collaboration in more detail, I propose a meeting next Tuesday, January 15th, at 10:00 AM. Would it be possible to meet at your office to have this discussion?

Thank you.

Best regards,
John
Senior Executive Director
Bike Corporation
"""
```


```python
from itertools import chain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessageChunk
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template(
    "Please extract the important parts of the following email.\n\n{email_conversation}"
)

chain = prompt | llm | StrOutputParser()

answer = chain.stream({"email_conversation": email_conversation})


#  A function for real-time output (streaming)
def stream_response(response, return_output=False):
    """
    Streams the response from the AI model, processing and printing each chunk.

    This function iterates over each item in the 'response' iterable. If an item is an instance of AIMessageChunk, it extracts and prints the content.
    If the item is a string, it prints the string directly.
    Optionally, the function can return the concatenated string of all response chunks.

    Args:
    - response (iterable): An iterable of response chunks, which can be AIMessageChunk objects or strings.
    - return_output (bool, optional): If True, the function returns the concatenated response string. The default is False.

    Returns:
    - str: If `return_output` is True, the concatenated response string. Otherwise, nothing is returned.
    """
    answer = ""
    for token in response:
        if isinstance(token, AIMessageChunk):
            answer += token.content
            print(token.content, end="", flush=True)
        elif isinstance(token, str):
            answer += token
            print(token, end="", flush=True)
    if return_output:
        return answer


output = stream_response(answer, return_output=True)
```

    ### Important Parts of the Email:
    
    - **From:** John (John@bikecorporation.me)
    - **To:** Kim (Kim@teddyinternational.me)
    - **Subject:** "ZENESIS" bike distribution cooperation and meeting schedule proposal
    
    - **Key Points:**
      - John is the Senior Executive Director at Bike Corporation.
      - He learned about the "ZENESIS" bicycle model through a press release.
      - Bike Corporation is a leading company in bicycle manufacturing and distribution.
      - They are requesting a detailed brochure for the ZENESIS model, specifically needing information on technical specifications, battery performance, and design aspects.
      - A meeting is proposed for Tuesday, January 15th, at 10:00 AM at Kim's office to discuss collaboration possibilities in more detail.
    
    - **Proposed Meeting:**
      - Date: Tuesday, January 15th
      - Time: 10:00 AM
      - Location: Kim's office
    
    - **Purpose:**
      - To discuss the possibilities for collaboration and further refine the distribution strategy and marketing plan for the ZENESIS model.

当不使用 output parser(PydanticOutputParser) 时，需要对数据类型和访问方式自定义


```python
answer = chain.invoke({"email_conversation": email_conversation})
print(answer)
```

    ### Important Parts of the Email:
    
    - **From:** John (John@bikecorporation.me)
    - **To:** Kim (Kim@teddyinternational.me)
    - **Subject:** "ZENESIS" bike distribution cooperation and meeting schedule proposal
    
    - **Key Points:**
      - John is the Senior Executive Director at Bike Corporation.
      - He learned about the "ZENESIS" bicycle model through a press release.
      - Bike Corporation is a leading company in bicycle manufacturing and distribution.
      - They are requesting a detailed brochure for the ZENESIS model, specifically needing information on technical specifications, battery performance, and design aspects.
      - A meeting is proposed for Tuesday, January 15th, at 10:00 AM at Kim's office to discuss collaboration possibilities in more detail.
    
    - **Proposed Meeting:**
      - Date: Tuesday, January 15th
      - Time: 10:00 AM
      - Location: Kim's office
    
    - **Follow-Up:**
      - John requests a detailed brochure for the ZENESIS model.
      - He is interested in discussing potential distribution and marketing strategies.


## 使用 PydanticOutputParser  
当提供类似上述的电子邮件内容时，我们将使用以下以 Pydantic 风格定义的类来解析邮件信息。  

作为参考， Field  内的  description  用于指导从文本响应中提取关键信息。LLM 依赖此描述来提取所需信息。因此，确保该描述准确且清晰至关重要。


```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class EmailSummary(BaseModel):
    person: str = Field(description="The sender of the email")
    email: str = Field(description="The email address of the sender")
    subject: str = Field(description="The subject of the email")
    summary: str = Field(description="A summary of the email content")
    date: str = Field(
        description="The meeting date and time mentioned in the email content"
    )


# Create PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=EmailSummary)
```


```python
print(parser.get_format_instructions())
```

    The output should be formatted as a JSON instance that conforms to the JSON schema below.
    
    As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
    the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
    
    Here is the output schema:
    ```
    {"properties": {"person": {"description": "The sender of the email", "title": "Person", "type": "string"}, "email": {"description": "The email address of the sender", "title": "Email", "type": "string"}, "subject": {"description": "The subject of the email", "title": "Subject", "type": "string"}, "summary": {"description": "A summary of the email content", "title": "Summary", "type": "string"}, "date": {"description": "The meeting date and time mentioned in the email content", "title": "Date", "type": "string"}}, "required": ["person", "email", "subject", "summary", "date"]}
    ```


我们还没有解析，这只是一个解析的说明。注意

```
{"properties": {"person": {"description": "The sender of the email", "title": "Person", "type": "string"}, "email": {"description": "The email address of the sender", "title": "Email", "type": "string"}, "subject": {"description": "The subject of the email", "title": "Subject", "type": "string"}, "summary": {"description": "A summary of the email content", "title": "Summary", "type": "string"}, "date": {"description": "The meeting date and time mentioned in the email content", "title": "Date", "type": "string"}}, "required": ["person", "email", "subject", "summary", "date"]}
```
这里交代了 parser.pydantic_object 即 EmailSummary 的参数名、数据类型、说明。

接下来可在 prompt 中对输出加上要求

定义提示：  

1. `question`：接收用户的问题。  
2. `email_conversation`：输入电子邮件对话的内容。  
3. `format`：指定格式。


```python
prompt = PromptTemplate.from_template(
    """
You are a helpful assistant. 

QUESTION:
{question}

EMAIL CONVERSATION:
{email_conversation}

FORMAT:
{format}
"""
)

# Add partial formatting of PydanticOutputParser to format
prompt = prompt.partial(format=parser.get_format_instructions())
```


```python
chain = prompt | llm

response = chain.stream(
    {
        "email_conversation": email_conversation,
        "question": "Extract the main content of the email.",
    }
)

# The result is provided in JSON format.
output = stream_response(response, return_output=True)
```

    ```json
    {
      "person": "John",
      "email": "John@bikecorporation.me",
      "subject": "ZENESIS bike distribution cooperation and meeting schedule proposal",
      "summary": "John, representing Bike Corporation, is interested in the ZENESIS bicycle model and requests a detailed brochure with technical specifications, battery performance, and design aspects. He proposes a meeting on January 15th, 2024, at 10:00 AM to discuss the possibilities for collaboration.",
      "date": "January 15th, 2024, at 10:00 AM"
    }
    ```

可以看出大模型按照要求输出了一个 json , 之后可用 `parser.parse()` 解析


```python
structured_output = parser.parse(output)
structured_output
```




    EmailSummary(person='John', email='John@bikecorporation.me', subject='ZENESIS bike distribution cooperation and meeting schedule proposal', summary='John, representing Bike Corporation, is interested in the ZENESIS bicycle model and requests a detailed brochure with technical specifications, battery performance, and design aspects. He proposes a meeting on January 15th, 2024, at 10:00 AM to discuss the possibilities for collaboration.', date='January 15th, 2024, at 10:00 AM')



parser 应该正常接在 chain 的后面


```python
# Reconstruct the entire chain by adding an output parser.
chain = prompt | llm | parser
# Execute the chain and print the results.
response = chain.invoke(
    {
        "email_conversation": email_conversation,
        "question": "Extract the main content of the email.",
    }
)

# The results are output in the form of an EmailSummary object.
print(response)
```

    person='John' email='John@bikecorporation.me' subject='ZENESIS bike distribution cooperation and meeting schedule proposal' summary="John, representing Bike Corporation, is interested in the ZENESIS bicycle model and requests a detailed brochure with technical specifications, battery performance, and design aspects. He proposes a meeting on January 15th, 2024, at 10:00 AM at Kim's office to discuss collaboration possibilities." date='January 15th, 2024, at 10:00 AM'


## with_structured_output(Pydantic)

通过使用 `.with_structured_output(Pydantic)`，您可以添加输出解析器并将输出转换为 Pydantic 对象。但他本身不支持流式生成


```python
llm_with_structured = llm.with_structured_output(EmailSummary)
answer = llm_with_structured.invoke(email_conversation)
answer
```




    EmailSummary(person='John', email='John@bikecorporation.me', subject='ZENESIS bike distribution cooperation and meeting schedule proposal', summary="John, from Bike Corporation, is interested in the ZENESIS bicycle model and would like a detailed brochure with technical specifications, battery performance, and design aspects. He proposes a meeting on January 15th at 10:00 AM at Kim's office to discuss the possibilities for collaboration.", date='January 15th, 10:00 AM')



# CommaSeparatedListOutputParser



`CommaSeparatedListOutputParser` 是 LangChain 中专门用于生成逗号分隔列表形式的结构化输出的解析器。  

它简化了数据提取和呈现的过程，使信息以清晰、简洁的列表格式呈现，特别适用于组织数据点、名称、项目或其他结构化值。通过利用此解析器，用户可以提高数据清晰度，确保格式一致，并改善工作流程效率，尤其是在需要结构化输出的应用场景中。


```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# Initialize the output parser
output_parser = CommaSeparatedListOutputParser()

# Retrieve format instructions for the output parser
format_instructions = output_parser.get_format_instructions()
print(format_instructions)
```

    Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`



```python
from langchain_core.prompts import PromptTemplate

# Define the prompt template
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],  # 'subject' will be dynamically replaced
    partial_variables={
        "format_instructions": format_instructions
    },  # Use parser's format instructions
)
print(prompt)
```

    input_variables=['subject'] input_types={} partial_variables={'format_instructions': 'Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`'} template='List five {subject}.\n{format_instructions}'



```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)
```


```python
chain = prompt | llm | output_parser

# Run the chain with a specific subject
result = chain.invoke({"subject": "famous landmarks in South Korea"})
print(result)

```

    ['Gyeongbokgung Palace', 'N Seoul Tower', 'Bukchon Hanok Village', 'Seoraksan National Park', 'Gwangjang Market']


### 使用 Python 索引访问数据  

由于 `CommaSeparatedListOutputParser` 会自动将输出格式化为 Python 列表，您可以通过索引轻松访问各个元素。


```python

print("First Landmark:", result[0])
print("Second Landmark:", result[1])
print("Last Landmark:", result[-1])
```

# StructuredOutputParser



`StructuredOutputParser` 是一种有价值的工具，可用于将大型语言模型（LLM）的响应格式化为字典结构，使多个字段以键/值对的形式返回。  

尽管 Pydantic 和 JSON 解析器提供了强大的功能，`StructuredOutputParser` 在处理较弱的模型（如参数较少的本地模型）时特别有效。对于智能水平低于 GPT 或 Claude 等高级模型的情况，它尤为有用。  

通过使用 `StructuredOutputParser`，开发者可以在各种 LLM 应用中保持数据的完整性和一致性，即使是在参数较少的模型上运行时也能确保稳定的输出。

### 使用 ResponseSchema 与 StructuredOutputParser  

- 使用 `ResponseSchema` 类定义响应模式，以包含用户问题的答案以及所使用的来源（网站）描述。  
- 使用 `response_schemas` 初始化 `StructuredOutputParser`，使输出符合定义的响应模式。  

**[注意]**  
在使用本地模型时，Pydantic 解析器可能经常无法正常工作。在这种情况下，`StructuredOutputParser` 是一个不错的替代方案。


```python
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Response to the user's question
response_schemas = [
    ResponseSchema(name="answer", description="Answer to the user's question"),
    ResponseSchema(
        name="source",
        description="The `source` used to answer the user's question, which should be a `website URL`.",
    ),
]
# Initialize the structured output parser based on the response schemas
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

```


```python
from langchain_core.prompts import PromptTemplate
# Parse the format instructions.
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    # Set up the template to answer the user's question as best as possible.
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    # Use 'question' as the input variable.
    input_variables=["question"],
    # Use 'format_instructions' as a partial variable.
    partial_variables={"format_instructions": format_instructions},
)
```


```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)

chain = prompt | llm | output_parser  # Connect the prompt, model, and output parser

# Ask the question, "What is the largest desert in the world?"
chain.invoke({"question": "What is the largest desert in the world?"})
```




    {'answer': 'The largest desert in the world is the Sahara Desert, which covers parts of North Africa.',
     'source': 'https://www.nationalgeographic.org/encyclopedia/sahara-desert/'}



### **PydanticOutputParser vs. StructuredOutputParser**

这两种解析器都是 LangChain 提供的工具，用于格式化和解析 LLM（大语言模型）的输出，但它们的适用场景和实现方式有所不同。以下是它们的主要区别：

| **特性**                 | **PydanticOutputParser**                                      | **StructuredOutputParser**                                      |
|-------------------------|-------------------------------------------------|--------------------------------------------------|
| **数据格式**            | 基于 **Pydantic** 数据模型，返回 Pydantic 对象  | 返回 **字典（dict）** 结构，适用于较简单的数据解析 |
| **适用场景**            | 适用于 **强大 LLM（如 GPT-4、Claude）**，保证数据结构一致 | 适用于 **较弱的 LLM（如本地小模型）**，对模型要求较低 |
| **解析方式**            | 使用 Pydantic 的数据验证机制，自动解析数据 | 通过自定义 `ResponseSchema` 定义键值对结构 |
| **错误容忍度**          | 对格式错误较为严格，格式不符会报错 | 宽松容错，适用于结构化要求较低的场景 |
| **是否支持复杂验证**    | ✅ **支持**，可定义字段类型、校验规则 | ❌ **不支持**，只能按键值结构输出 |
| **是否适合本地模型**    | ❌ **不太适合**，因本地模型输出不稳定，容易解析失败 | ✅ **适合**，能适应本地小模型的不稳定输出 |

注意 PydanticOutputParser 解析为 pydantic.BaseModel 类实例, StructuredOutputParser 为 Dict. 

### **总结**
- **PydanticOutputParser**：适用于复杂结构和数据校验，适合强大 LLM（GPT-4、Claude）。
- **StructuredOutputParser**：适用于本地小模型，结构简单但更灵活。

# JsonOutputParser



`JsonOutputParser` 是一个工具，允许用户指定所需的 JSON 模式。它旨在使大型语言模型（LLM）能够查询数据并以符合指定模式的 JSON 格式返回结果。  

为了确保 LLM 能够准确高效地处理数据，并生成符合要求的 JSON 格式，模型必须具备足够的计算能力（如智能水平）。例如，`llama-70B` 模型比 `llama-8B` 模型具有更大的计算能力，因此更适合处理复杂数据。  


**JSON（JavaScript Object Notation）** 是一种轻量级的数据交换格式，用于存储和组织数据。它在 Web 开发中至关重要，并广泛用于服务器与客户端之间的通信。JSON 基于文本，易读且易于机器解析和生成。  

JSON 数据由**键值对（key-value pairs）**组成，其中 "key" 是字符串，而 "value" 可以是多种数据类型。JSON 主要有两种基本结构：  

- **对象（Object）**：由大括号 `{ }` 包围的一组键值对。每个键和值之间用冒号（`:`）分隔，多个键值对之间用逗号（`,`）分隔。  
- **数组（Array）**：由方括号 `[ ]` 包围的有序值列表，数组内的值之间用逗号（`,`）分隔。  

```json
{
  "name": "John Doe",
  "age": 30,
  "is_student": false,
  "skills": ["Java", "Python", "JavaScript"],
  "address": {
    "street": "123 Main St",
    "city": "Anytown"
  }
}
```

## 1. 使用 Pydantic

类似 PydanticOutputParser 的创建方式, 还是需要创建 BaseModel, 但解析为 Dict


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)

# Use Pydantic to define the data schema for the output format.
class Topic(BaseModel):
    description: str = Field(description="A concise description of the topic")
    hashtags: str = Field(description="Keywords in hashtag format (at least 2)")

# Write your question
question = "Please explain the severity of global warming."

# Set up the parser and inject the instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=Topic)

# Set up the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly AI assistant. Answer questions concisely."),
        ("user", "#Format: {format_instructions}\n\n#Question: {question}"),
    ]
)

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Combine the prompt, model, and JsonOutputParser into a chain
chain = prompt | llm | parser

# Run the chain with your question
answer = chain.invoke({"question": question})

print(answer)
print(type(answer))
```

    {'description': 'Global warming poses severe threats to ecosystems, human health, and economic stability. It leads to rising sea levels, more frequent extreme weather events, and disruptions to agricultural productivity. The consequences are far-reaching, affecting biodiversity, water resources, and global food security.', 'hashtags': '#globalwarming #climatechange #severethreats #ecosystemdisruption'}
    <class 'dict'>


## 2. 不使用 Pydantic

其实对应的 format instruction 只是简单的说明: 'Return a JSON object.'. 因此你还需要在 prompt 的其他部分加上输出元素的说明.


```python
# Write your question
question = "Please provide information about global warming. Include the explanation in description and the related keywords in `hashtags`."

# Initialize JsonOutputParser
parser = JsonOutputParser()

# Set up the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly AI assistant. Answer questions concisely."),
        ("user", "#Format: {format_instructions}\n\n#Question: {question}"),
    ]
)

# Inject instruction to prompt
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Combine the prompt, model, and JsonOutputParser into a chain
chain = prompt | llm | parser

# Run the chain with your question
response = chain.invoke({"question": question})
print(response)
```

    {'description': "Global warming refers to the long-term increase in Earth's average surface temperature due to human activities, primarily the emission of greenhouse gases like carbon dioxide and methane. This leads to various environmental impacts such as rising sea levels, more frequent extreme weather events, and loss of biodiversity. Mitigation strategies include reducing carbon emissions, increasing use of renewable energy, and promoting sustainable practices.", 'hashtags': ['globalwarming', 'climatechange', 'greenhousegases', 'carbonfootprint', 'renewableenergy', 'sustainability', 'environmentalimpact']}



```python
parser.get_format_instructions()
```




    'Return a JSON object.'



# DatetimeOutputParser



`DatetimeOutputParser` 是一个输出解析器，用于生成 **`datetime` 对象** 形式的结构化输出。  

通过将 LLM 的输出转换为 `datetime` 对象，它可以实现更**系统化和一致性**的日期和时间数据处理，使其在数据处理和分析中非常有用。  

如果需要以日期或时间的形式生成输出，LangChain 提供的 `DatetimeOutputParser` 可以简化这一过程。  

`DatetimeOutputParser` 的 **`format`（格式）** 可以按照下表中的格式代码进行指定：  

| **格式代码** | **描述**                 | **示例**                  |  
|-------------|-------------------------|---------------------------|  
| `%Y`        | 4 位数年份               | 2024                      |  
| `%y`        | 2 位数年份               | 24                        |  
| `%m`        | 2 位数月份               | 07                        |  
| `%d`        | 2 位数日期               | 04                        |  
| `%H`        | 24 小时制小时            | 14                        |  
| `%I`        | 12 小时制小时            | 02                        |  
| `%p`        | AM 或 PM                 | PM                        |  
| `%M`        | 2 位数分钟               | 45                        |  
| `%S`        | 2 位数秒                 | 08                        |  
| `%f`        | 6 位数微秒               | 000123                    |  
| `%z`        | UTC 偏移量               | +0900                     |  
| `%Z`        | 时区名称                 | KST                       |  
| `%a`        | 缩写星期名               | Thu                       |  
| `%A`        | 完整星期名               | Thursday                  |  
| `%b`        | 缩写月份名               | Jul                       |  
| `%B`        | 完整月份名               | July                      |  
| `%c`        | 完整日期和时间           | Thu Jul 4 14:45:08 2024   |  
| `%x`        | 完整日期                 | 07/04/24                  |  
| `%X`        | 完整时间                 | 14:45:08                  |  


```python
from langchain.output_parsers import DatetimeOutputParser
from langchain_core.prompts import PromptTemplate

# Initialize the output parser
# format 参数是输入数据的格式, 会转换为 Datetime 实例
output_parser = DatetimeOutputParser(format="%Y-%m-%d")

# Get format instructions
format_instructions = output_parser.get_format_instructions()

# Create answer template for user questions
template = """Answer the users question:\n\n#Format Instructions: \n{format_instructions}\n\n#Question: \n{question}\n\n#Answer:"""

# Create a prompt from the template
prompt = PromptTemplate.from_template(
    template,
    partial_variables={
        "format_instructions": format_instructions,
    },  # Use parser's format instructions
)

print(format_instructions)
print("-----------------------------------------------\n")
print(prompt)
```

    Write a datetime string that matches the following pattern: '%Y-%m-%d'.
    
    Examples: 1947-12-08, 1650-03-11, 177-12-05
    
    Return ONLY this string, no other words!
    -----------------------------------------------
    
    input_variables=['question'] input_types={} partial_variables={'format_instructions': "Write a datetime string that matches the following pattern: '%Y-%m-%d'.\n\nExamples: 1947-12-08, 1650-03-11, 177-12-05\n\nReturn ONLY this string, no other words!"} template='Answer the users question:\n\n#Format Instructions: \n{format_instructions}\n\n#Question: \n{question}\n\n#Answer:'



```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)

chain = prompt | llm | output_parser

# Call the chain to get an answer to the question
output = chain.invoke({"question": "The year Google was founded"})

print(output)
print(type(output))
```

    1998-09-04 00:00:00
    <class 'datetime.datetime'>


# EnumOutputParser



`EnumOutputParser` 是一个工具，用于严格将语言模型的输出解析为**预定义的枚举（Enum）值**，确保模型输出始终是枚举值之一，并具有以下特点：  

- **枚举解析**：将字符串输出转换为预定义的 `Enum` 值。  
- **类型安全**：确保解析结果始终是定义的 `Enum` 值之一。  
- **灵活性**：自动处理空格和换行符。

**应用场景**  
- 当你只希望从一组可能的选项中选择一个有效值时。  
- 通过使用明确的 Enum 值，避免拼写错误和不一致的变体。  

在下面的示例中，我们定义了一个 `Colors` 枚举，并通过解析输出，使 LLM 只能返回 `red`、`green` 或 `blue` 其中之一。


```python
from langchain.output_parsers.enum import EnumOutputParser

from enum import Enum

class Colors(Enum):
    RED = "Red"
    GREEN = "Green"
    BLUE = "Blue"

# Instantiate EnumOutputParser
parser = EnumOutputParser(enum=Colors)

# You can view the format instructions that the parser expects.
print(parser.get_format_instructions())
```

    Select one of the following options: Red, Green, Blue



```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)

# Prompt template: the parser's format instructions are added at the end.
prompt = (
    PromptTemplate.from_template(
        """Which color is this object?

Object: {object}

Instructions: {instructions}"""
    ).partial(instructions=parser.get_format_instructions())
)

# Entire chain: (prompt) -> (LLM) -> (Enum Parser)
chain = prompt | llm | parser

response = chain.invoke({"object": "sky"})
print("Parsed Enum:", response)
print("Raw Enum Value:", response.value)
```

    Parsed Enum: Colors.BLUE
    Raw Enum Value: Blue


如果模型输出的不是 Enum 值，解析会报错

# OutputFixingParser



`OutputFixingParser` 是 LangChain 提供的一个自动化机制，用于**修正解析过程中可能出现的错误**。该解析器可以包装其他解析器（如 `PydanticOutputParser`），当底层解析器遇到**格式错误或不符合预期格式**的输出时，它会**自动介入**，并利用额外的 LLM 调用来修正错误，确保输出符合正确的格式。  

`OutputFixingParser` 主要用于处理**初始输出不符合预定义模式**的情况。如果出现问题，解析器会**自动检测格式错误**，并向模型提交新的请求，包含**具体的错误修正指令**。这些指令明确指出问题所在，并提供清晰的修正指导，以确保数据格式符合预期要求。  

这一功能在**严格遵循数据模式的应用场景**中非常有用。例如，在使用 `PydanticOutputParser` 生成符合特定数据模式的输出时，可能会遇到**缺失字段或数据类型错误**的问题。  

**`OutputFixingParser` 的工作流程**
1. **错误检测**：识别输出不符合预定义的数据模式。  
2. **错误修正**：向 LLM 生成新的请求，并提供明确的修正指令。  
3. **重新格式化输出**：确保修正指令**精准定位错误**（如缺少字段、数据类型错误等），并指导 LLM 重新格式化输出，使其符合预期数据模式。  




假设你正在使用 `PydanticOutputParser` 来强制执行一个模式，该模式要求输出包含以下字段：
- `name`（字符串）
- `age`（整数）
- `email`（字符串）

如果 LLM 生成的输出中：
- **缺少 `age` 字段**
- **`email` 不是有效的字符串格式**

那么 `OutputFixingParser` 会自动介入，并向 LLM 发送新的请求，例如：
- `"输出缺少 `age` 字段，请添加一个适当的整数值。" `
- `"`email` 字段的格式无效，请修正为有效的电子邮件地址格式。"`  

这一**迭代式纠正过程**确保最终输出符合指定的数据模式，无需人工干预。  



```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


# Define the Actor class using Pydantic
class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(description="list of names of films they starred in")


# A query to generate the filmography for a random actor
actor_query = "Generate the filmography for a random actor."

# Use PydanticOutputParser to parse the output into an Actor object
parser = PydanticOutputParser(pydantic_object=Actor)
```

### 尝试解析格式错误的输入数据

- **格式错误的变量**包含一个格式不正确的字符串，该字符串与预期的结构不匹配（例如使用了单引号 `'` 而不是双引号 `"`）。
- 调用 `parser.parse()` 时，由于格式不匹配，系统会抛出错误。


```python
misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"
parser.parse(misformatted)
```


    ---------------------------------------------------------------------------

    JSONDecodeError                           Traceback (most recent call last)

    File /data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/site-packages/langchain_core/output_parsers/json.py:83, in JsonOutputParser.parse_result(self, result, partial)
         82 try:
    ---> 83     return parse_json_markdown(text)
         84 except JSONDecodeError as e:


    File /data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/site-packages/langchain_core/utils/json.py:144, in parse_json_markdown(json_string, parser)
        143     json_str = json_string if match is None else match.group(2)
    --> 144 return _parse_json(json_str, parser=parser)


    File /data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/site-packages/langchain_core/utils/json.py:160, in _parse_json(json_str, parser)
        159 # Parse the JSON string into a Python dictionary
    --> 160 return parser(json_str)


    File /data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/site-packages/langchain_core/utils/json.py:118, in parse_partial_json(s, strict)
        115 # If we got here, we ran out of characters to remove
        116 # and still couldn't parse the string as JSON, so return the parse error
        117 # for the original string.
    --> 118 return json.loads(s, strict=strict)


    File /data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/json/__init__.py:359, in loads(s, cls, object_hook, parse_float, parse_int, parse_constant, object_pairs_hook, **kw)
        358     kw['parse_constant'] = parse_constant
    --> 359 return cls(**kw).decode(s)


    File /data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/json/decoder.py:337, in JSONDecoder.decode(self, s, _w)
        333 """Return the Python representation of ``s`` (a ``str`` instance
        334 containing a JSON document).
        335 
        336 """
    --> 337 obj, end = self.raw_decode(s, idx=_w(s, 0).end())
        338 end = _w(s, end).end()


    File /data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/json/decoder.py:353, in JSONDecoder.raw_decode(self, s, idx)
        352 try:
    --> 353     obj, end = self.scan_once(s, idx)
        354 except StopIteration as err:


    JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)

    
    The above exception was the direct cause of the following exception:


    OutputParserException                     Traceback (most recent call last)

    Cell In[34], line 2
          1 misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"
    ----> 2 parser.parse(misformatted)


    File /data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/site-packages/langchain_core/output_parsers/pydantic.py:83, in PydanticOutputParser.parse(self, text)
         74 def parse(self, text: str) -> TBaseModel:
         75     """Parse the output of an LLM call to a pydantic object.
         76 
         77     Args:
       (...)
         81         The parsed pydantic object.
         82     """
    ---> 83     return super().parse(text)


    File /data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/site-packages/langchain_core/output_parsers/json.py:97, in JsonOutputParser.parse(self, text)
         88 def parse(self, text: str) -> Any:
         89     """Parse the output of an LLM call to a JSON object.
         90 
         91     Args:
       (...)
         95         The parsed JSON object.
         96     """
    ---> 97     return self.parse_result([Generation(text=text)])


    File /data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/site-packages/langchain_core/output_parsers/pydantic.py:72, in PydanticOutputParser.parse_result(self, result, partial)
         70 if partial:
         71     return None
    ---> 72 raise e


    File /data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/site-packages/langchain_core/output_parsers/pydantic.py:67, in PydanticOutputParser.parse_result(self, result, partial)
         54 """Parse the result of an LLM call to a pydantic object.
         55 
         56 Args:
       (...)
         64     The parsed pydantic object.
         65 """
         66 try:
    ---> 67     json_object = super().parse_result(result)
         68     return self._parse_obj(json_object)
         69 except OutputParserException as e:


    File /data02/hyzhang10/miniconda3/envs/xp-nlp/lib/python3.12/site-packages/langchain_core/output_parsers/json.py:86, in JsonOutputParser.parse_result(self, result, partial)
         84 except JSONDecodeError as e:
         85     msg = f"Invalid json output: {text}"
    ---> 86     raise OutputParserException(msg, llm_output=text) from e


    OutputParserException: Invalid json output: {'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}
    For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/OUTPUT_PARSING_FAILURE 


## 使用 OutputFixingParser 修正格式错误  
### 设置 OutputFixingParser 自动修正错误  
- `OutputFixingParser` 包装了现有的 `PydanticOutputParser`，通过向 LLM 发送额外请求，自动修正错误。  
- `from_llm()` 方法将 `OutputFixingParser` 与 `ChatOpenAI` 连接，以修正输出中的格式问题。


```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import OutputFixingParser

llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)


# Define a custom prompt to provide the fixing instructions
fixing_prompt = PromptTemplate(
    template=(
        "The following JSON is incorrectly formatted or incomplete: {completion}\n"
    ),
    input_variables=[
        "completion",
    ],
)

# Use OutputFixingParser to automatically fix the error
new_parser = OutputFixingParser.from_llm(
    parser=parser, llm=llm, prompt=fixing_prompt
)
```

### 使用 OutputFixingParser 解析格式错误的输出  
- 使用 `new_parser.parse()` 方法解析格式错误的数据。  
- `OutputFixingParser` 将自动修正数据中的错误，并生成一个有效的 `Actor` 对象。


```python
# Attempt to parse the misformatted JSON with Exception Handling
try:
    actor = new_parser.parse(misformatted)
    print("Parsed actor:", actor)
except Exception as e:
    print("Error while parsing:", e)
```

    Parsed actor: name='Tom Hanks' film_names=['Forrest Gump', 'Cast Away', 'Saving Private Ryan']


**✅ `OutputFixingParser` 能做什么**
- **修复格式错误**（如 JSON 结构错误、单引号替换为双引号等）。
- **修正轻微的数据错误**（如字符串拼写错误、数字格式调整等）。
- **依赖 LLM 进行数据填充**（如果 LLM 逻辑足够强大，可能会补全缺失字段）。

**❌ `OutputFixingParser` 不能自动推理缺失变量的正确值**
如果**初始 LLM 输出中缺少某些变量**（如 `age` 字段缺失），但 **修复过程并不会参考原始 Prompt**，那么 `OutputFixingParser` 可能不会自动填充正确的数据，而是直接报错或返回 `None`。

**为什么？**
1. **`OutputFixingParser` 只是对 LLM 进行二次调用**，它的主要任务是**格式修正**，而不是内容推理。
2. **它不会“记住”原始的 Prompt**，所以无法基于上下文填充**缺失的字段**（除非 LLM 自行推理）。
3. **LLM 可能不会自动填充缺失字段**，除非你明确告诉它如何修正数据。

你可以通过改进 fixing_prompt，让 LLM 在修复过程中不仅修正格式，还填补缺失的变量。


```python
fixing_prompt = PromptTemplate(
    template=(
		"Question:\n{question}\n"
        "Format:\n {format}"
		"Answer:\n{answer}\n"
        "The above answer is incorrectly formatted or incomplete.\n"
		"Correct any format mistake. "
		"If any required field is missing, generate a reasonable value according to the question.\n"
    ),
    input_variables=["completion"],
	partial_variables={
		"format":parser.get_format_instructions
	}
)
```

# 初始化大模型接口并调用

一般都是用ChatOpenAI这个类，以下两个方式引用都可以:
- from langchain_openai.chat_models import ChatOpenAI
- from langchain_openai import ChatOpenAI 


这是对话的重要参数,关键的是model.
Key init args — completion params:

- model: str, Name of OpenAI model to use.
- temperature: float, Sampling temperature.
- max_tokens: Optional[int], Max number of tokens to generate.
- logprobs: Optional[bool], Whether to return logprobs.
- stream_options: Dict, Configure streaming outputs, like whether to return token usage when streaming (``{"include_usage": True}``).

这是客户端的重要参数，关键的有base_url和api_key.
Key init args — client params:

- timeout: Union[float, Tuple[float, float], Any, None], Timeout for requests.
- max_retries: int, Max number of retries.
- api_key: Optional[str], OpenAI API key. If not passed in will be read from env var OPENAI_API_KEY.
- base_url: Optional[str], Base URL for API requests. Only specify if using a proxy or service


```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)

```

使用`invoke`函数来调用大模型接口


```python
question = "What is the capital of USA?"

llm.invoke(question)
```

### 响应格式（AIMessage类）

在使用 `ChatOpenAI` 对象时，响应以 **AI 消息** 的格式返回。这包括模型生成的文本内容以及与响应相关的元数据或附加属性。这些信息提供了关于 AI 回复的结构化数据，以及响应的生成方式。

**AI 消息的关键组成部分**
1. **`content`**  
   - **定义：** 由 AI 生成的主要响应文本。  
   - **示例：** **"韩国的首都是首尔。"**  
   - **作用：** 这是用户与 AI 交互的主要部分。

2. **`response_metadata`**  
   - **定义：** 关于响应生成过程的元数据。  
   - **主要字段：**  
     - **`model_name` ：** 使用的模型名称（例如 `"gpt-4o-mini"`）。  
     - **`finish_reason` ：** 生成停止的原因（`stop` 表示正常完成）。  
     - **`token_usage` ：** 令牌使用详情：
       - **`prompt_tokens` ：** 输入查询使用的令牌数。  
       - **`completion_tokens` ：** 响应内容使用的令牌数。  
       - **`total_tokens` ：** 输入和输出的总令牌数。

3. **`id`**  
   - **定义：** API 调用的唯一标识符。  
   - **作用：** 便于跟踪或调试特定交互。



```python
# 示例
"""
AIMessage(
    content='The capital of the United States is Washington, D.C.', 
    additional_kwargs={
        'refusal': None
    }, 
    response_metadata={
        'token_usage': {
            'completion_tokens': 13, 
            'prompt_tokens': 36, 
            'total_tokens': 49, 
            'completion_tokens_details': None, 
            'prompt_tokens_details': None
        }, 
        'model_name': 'Qwen2.5-7B-Instruct', 
        'system_fingerprint': None, 
        'finish_reason': 'stop', 
        'logprobs': None}, 
    id='run-e2adb89c-7c83-4a53-b68a-be914308c468-0', 
    usage_metadata={
        'input_tokens': 36, 
        'output_tokens': 13, 
        'total_tokens': 49, 
        'input_token_details': {}, 
        'output_token_details': {}
    }
)
"""
```

### **流式输出**  

流式选项特别适用于接收查询的实时响应。  
与等待整个响应生成完成不同，该模型会逐个令牌或按数据块流式传输输出，从而实现更快的交互和即时反馈。


```python
answer = llm.stream(
    "Please provide 10 beautiful tourist destinations in USA along with their addresses!"
)

# 这种流式生成方式本质上是 迭代器 (iterator) 的一种应用。
for token in answer:
    print(token.content, end="", flush=True)
```

# **链式创建（Chain Creation）**  


在这里，我们使用 **LCEL（LangChain Expression Language / LangChain 表达式语言）** 将多个组件组合成一个完整的链。  

```
chain = prompt | model | output_parser
```

- **`|` 运算符** 类似于 [Unix 管道操作符](<https://en.wikipedia.org/wiki/Pipeline_(Unix)>)，用于连接不同的组件，并将一个组件的输出作为下一个组件的输入。  

在这个链式结构中：
1. 用户输入被传递到 **提示模板（PromptTemplate）**。  
2. **提示模板** 处理输入并生成结构化的提示。  
3. **模型（LLM）** 接收提示并生成响应。  
4. **输出解析器（Output Parser）** 进一步解析并格式化最终输出。  

通过单独检查每个组件，可以清楚地理解每一步的处理过程。

## **Prompt 模板**  

`PromptTemplate` 用于通过用户的输入变量创建完整的提示字符串。
- **`template`**：模板字符串是一个预定义的格式，其中使用大括号 `{}` 表示变量。  
- **`input_variables`**：以列表形式定义要插入到大括号 `{}` 中的变量名称。  


```python
from langchain_core.prompts import PromptTemplate
# Define template
template = "What is the capital of {country}?"

# Create a `PromptTemplate` object using the `from_template` method.
prompt_template = PromptTemplate.from_template(template)
prompt_template
```


```python
prompt_template.format(country="Korea")
```

## **大模型接口**


```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)
model
```


```python
question = "What is the capital of USA?"
model.invoke(question)
```

## **输出解析器（Output Parser）**  

**输出解析器（Output Parser）** 是一种用于转换或处理 AI 模型响应的工具。由于模型的输出通常是 **自由格式文本（free-form text）**，因此 **输出解析器** 在以下方面至关重要：  
- **将输出转换为结构化格式**（如 JSON、表格或特定的数据结构）。  
- **提取所需的数据**，过滤无关信息，以便更高效地使用 AI 生成的内容。


```python
from langchain_core.output_parsers import StrOutputParser

# 直接返回str
output_parser = (
    StrOutputParser()
)
```

## 组成chain

**调用 `invoke()`**  
- 输入值(prompt模板中的变量)以 **Python 字典**（键值对）的形式提供。
- 在调用 `invoke()` 函数时，这些输入值作为参数传递。


```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template("Please explain {topic} in simple terms.")
model = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)
output_parser = StrOutputParser()
chain = prompt | model | output_parser
```


```python
input = {"topic": "The Principles of Learning in Artificial Intelligence Models"}
print(chain.invoke(input))
```


```python
# 同理，chain也可以流式生成

answer = chain.stream(input)

for token in answer:
    print(token, end="", flush=True)
```

    Sure! The principles of learning in artificial intelligence (AI) models can be explained in simple terms as follows:
    
    ### 1. **Data**
    - **What it is:** Data are the raw materials that AI models use to learn.
    - **Why it's important:** Good quality data helps the model understand patterns and make accurate predictions.
    
    ### 2. **Training**
    - **What it is:** Training is the process where the AI model learns from the data.
    - **How it works:** The model is shown examples (data) and adjusts its internal parameters to minimize errors in predictions.
    
    ### 3. **Model**
    - **What it is:** A model is the mathematical representation of the AI system.
    - **How it works:** It processes input data and produces output predictions or decisions.
    
    ### 4. **Parameters**
    - **What they are:** Parameters are the internal settings or weights of the model that are adjusted during training.
    - **Why they're important:** These settings determine how the model makes predictions.
    
    ### 5. **Loss Function**
    - **What it is:** A loss function measures how wrong the model's predictions are.
    - **How it works:** The model tries to minimize this function by adjusting its parameters.
    
    ### 6. **Optimization**
    - **What it is:** Optimization is the process of finding the best set of parameters to minimize the loss function.
    - **How it works:** Algorithms like gradient descent are used to iteratively adjust the parameters.
    
    ### 7. **Validation**
    - **What it is:** Validation is the process of checking how well the model performs on new, unseen data.
    - **Why it's important:** It helps ensure the model generalizes well to new data and isn't just memorizing the training data.
    
    ### 8. **Testing**
    - **What it is:** Testing is the final step where the model's performance is evaluated on a completely separate set of data.
    - **Why it's important:** It gives a final assessment of how well the model will perform in real-world scenarios.
    
    ### 9. **Feedback Loop**
    - **What it is:** A feedback loop is where the model's predictions are used to improve the training data or the model itself.
    - **Why it's important:** Continuous improvement can lead to better performance over time.
    
    ### 10. **Regularization**
    - **What it is:** Regularization is a technique to prevent overfitting by adding a penalty for complex models.
    - **Why it's important:** It helps the model generalize better to new data by avoiding overly complex solutions.
    
    ### 11. **Evaluation Metrics**
    - **What they are:** Evaluation metrics are specific measures used to assess the performance of the model.
    - **Examples:** Accuracy, precision, recall, F1 score, etc.
    - **Why they're important:** They provide a quantitative way to compare different models or evaluate the effectiveness of changes.
    
    ### 12. **Hyperparameters**
    - **What they are:** Hyperparameters are settings that control the training process but are not learned from the data.
    - **Examples:** Learning rate, batch size, number of layers in a neural network.
    - **Why they're important:** They can significantly affect the model's performance and need to be carefully tuned.
    
    These principles form the foundation of how AI models learn and improve over time. By understanding these concepts, you can better appreciate how AI systems are developed and deployed in various applications.

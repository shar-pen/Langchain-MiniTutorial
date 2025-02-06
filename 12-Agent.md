# Tool

工具是一个接口，允许代理、链条或大语言模型与外部世界互动。

LangChain 提供了易于使用的内置工具，并且还允许用户轻松构建自定义工具。

**你可以在下面的链接中找到集成到 LangChain 中的工具列表。**

- [集成到 LangChain 中的工具列表](https://python.langchain.com/docs/integrations/tools/)

## 内置工具 Built-in tools

你可以使用 LangChain 提供的预定义工具和工具包。

工具指的是单一的实用工具，而工具包将多个工具组合成一个单元供使用。

你可以在下面的链接中找到相关的工具。

**注意**
- [LangChain 工具/工具包](https://python.langchain.com/docs/integrations/tools/)

web 检索工具


```python
from langchain_community.utilities import GoogleSerperAPIWrapper

search = GoogleSerperAPIWrapper()
search.run("Obama's first name?")
```

图片生成工具


```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Initialize the ChatOpenAI model
llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)

# Define a prompt template for DALL-E image generation
prompt = PromptTemplate.from_template(
    "Generate a detailed IMAGE GENERATION prompt for DALL-E based on the following description. "
    "Return only the prompt, no intro, no explanation, no chatty, no markdown, no code block, no nothing. Just the prompt"
    "Output should be less than 1000 characters. Write in English only."
    "Image Description: \n{image_desc}",
)

# Create a chain connecting the prompt, LLM, and output parser
chain = prompt | llm | StrOutputParser()

# Execute the chain
image_prompt = chain.invoke(
    {"image_desc": "A Neo-Classicism painting satirizing people looking at their smartphones."}
)

# Output the image prompt
print(image_prompt)
```

几乎所有 tool 都是需要 api key 的

## Python REPL 工具

此工具提供一个类，用于在 **REPL (Read-Eval-Print Loop)** 环境中执行 Python 代码。
- [PythonREPLTool](https://python.langchain.com/docs/integrations/tools/python/)

**描述**

- 提供一个 Python shell 环境。
- 执行有效的 Python 命令作为输入。
- 使用 `print(...)` 函数查看结果。

**主要特点**

- sanitize_input：可选项，用于清理输入（默认：True）
- python_repl：**PythonREPL** 的实例（默认：在全局作用域中执行）

**使用方法**

- 创建 `PythonREPLTool` 的实例。
- 使用 `run`、`arun` 或 `invoke` 方法执行 Python 代码。

**输入清理**

- 从输入字符串中移除不必要的空格、反引号、关键字 "python" 和其他多余的元素。


```python
from langchain_experimental.tools import PythonREPLTool

# Creates a tool for executing Python code.
python_tool = PythonREPLTool()

# Executes Python code and returns the results.
print(python_tool.invoke("print(100 + 200)"))
```

下面是请求大语言模型编写 Python 代码并返回结果的示例。

**工作流程概述**
1. 请求大语言模型为特定任务编写 Python 代码。
2. 执行生成的代码以获取结果。
3. 输出结果。



```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

python_tool = PythonREPLTool()
# A function that executes Python code, outputs intermediate steps, and returns the tool execution results.
def print_and_execute(code, debug=True):
    if debug:
        print("CODE:")
        print(code)
    return python_tool.invoke(code)


# A prompt requesting Python code to be written.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Raymond Hetting, an expert python programmer, well versed in meta-programming and elegant, concise and short but well documented code. You follow the PEP8 style guide. "
            "Return only the code, no intro, no explanation, no chatty, no markdown, no code block, no nothing. Just the code.",
        ),
        ("human", "{input}"),
    ]
)
# Create LLM model.
llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)

# Create a chain using the prompt and the LLM model.
chain = prompt | llm | StrOutputParser() | RunnableLambda(print_and_execute)

# Outputting the results.
print(chain.invoke("Write code to generate Powerball numbers."))
```

## 自定义工具

除了 LangChain 提供的内置工具外，你还可以定义和使用自己的自定义工具。

为此，可以使用 `langchain.tools` 模块提供的 `@tool` 装饰器将一个函数转换为工具。

@tool 装饰器: 这个装饰器允许你将一个函数转换为工具。它提供了各种选项来定制工具的行为。

**使用方法**
1. 在函数上方应用 `@tool` 装饰器。
2. 根据需要设置装饰器参数。

使用这个装饰器，你可以轻松地将常规 Python 函数转换为强大的工具，从而实现自动化文档生成和灵活的接口创建。


```python
from langchain.tools import tool


# Convert a function into a tool using a decorator.
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

# Execute tool.
print(add_numbers.invoke({"a": 3, "b": 4}))
print(multiply_numbers.invoke({"a": 3, "b": 4}))
```

## 创建一个用于 Google 新闻文章搜索的自定义工具

定义 `GoogleNews` 类，该类将作为一个工具，用于搜索 Google 新闻文章。

**注意**
- 不需要 API 密钥（因为它使用 RSS 源）。

此工具用于搜索由 **news.google.com** 提供的新闻文章。

**描述**
- 使用 Google 新闻搜索 API 来检索最新的新闻。
- 允许基于关键词搜索新闻。

**主要参数**
- `k` (int)：返回的最大搜索结果数（默认：5）。

```python
# hl: 语言, gl: 区域, ceid: 区域和语言代码
url = f"{self.base_url}?hl=en&gl=US&ceid=US:en" 
```

在代码中，你可以通过修改语言 (hl)、区域 (gl) 和区域与语言代码 (ceid) 来调整搜索结果的语言和区域。

**注意**

将提供的代码保存为 `google_news.py`，然后你可以在其他文件中使用 `from google_news import GoogleNews` 进行导入。


```python
import feedparser
from urllib.parse import quote
from typing import List, Dict, Optional


class GoogleNews:
    """
    This is a class for searching Google News and returning the results.
    """

    def __init__(self):
        """
        Initializes the GoogleNews class.
        Sets the base_url attribute.
        """
        self.base_url = "https://news.google.com/rss"

    def _fetch_news(self, url: str, k: int = 3) -> List[Dict[str, str]]:
        """
        Fetches news from the given URL.

        Args:
            url (str): The URL to fetch the news from.
            k (int): The maximum number of news articles to fetch (default: 3).

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing news titles and links.
        """
        news_data = feedparser.parse(url)
        return [
            {"title": entry.title, "link": entry.link}
            for entry in news_data.entries[:k]
        ]

    def _collect_news(self, news_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Formats and returns the list of news articles.

        Args:
            news_list (List[Dict[str, str]]): A list of dictionaries containing news information.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing URLs and content.
        """
        if not news_list:
            print("No news available for the given keyword.")
            return []

        result = []
        for news in news_list:
            result.append({"url": news["link"], "content": news["title"]})

        return result

    def search_latest(self, k: int = 3) -> List[Dict[str, str]]:
        """
        Searches for the latest news.

        Args:
            k (int): The maximum number of news articles to search for (default: 3).

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing URLs and content.
        """
        #url = f"{self.base_url}?hl=ko&gl=KR&ceid=KR:ko"
        url = f"{self.base_url}?hl=en&gl=US&ceid=US:en" # hl: 언어, gl: 지역, ceid: 지역 및 언어 코드
        news_list = self._fetch_news(url, k)
        return self._collect_news(news_list)

    def search_by_keyword(
        self, keyword: Optional[str] = None, k: int = 3
    ) -> List[Dict[str, str]]:
        """
        Searches for news using a keyword.  

        Args:
            keyword (Optional[str]): The keyword to search for (default: None).
            k (int): The maximum number of news articles to search for (default: 3).

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing URLs and content.
        """
        if keyword:
            encoded_keyword = quote(keyword)
            url = f"{self.base_url}/search?q={encoded_keyword}"
        else:
            url = f"{self.base_url}?hl=en&gl=US&ceid=US:en"
        news_list = self._fetch_news(url, k)
        return self._collect_news(news_list)


google_tool = GoogleNews()
```


```python
google_tool.search_by_keyword("AI Investment")
```

地址是 google 的, 得翻墙, 以下是示例结果  
[{'url': 'https://news.google.com/rss/articles/CBMimAFBVV95cUxPNkFrLURMdEZWOV9zdmRrTUhNbVFkdWswZWx2Qmh4cTJlMmFIdmpsQ3doaVluenA3TEJaT0U3RWVmanl3TTQ5V3RfS3kyYVpydEloNWZXbjBmSF85MGR5cjNFSFI5eFhtTGdIVlNXX3UxNmxwMnVIb2NkTXA5WFVZR2hKLUw5RU9iT3k1Zno2UG10N2h1b2g5Sw?oc=5',
  'content': 'Nvidia Calls China\'s DeepSeek an "Excellent AI Advancement": Should Investors Press the Buy Button? - The Motley Fool'},
 {'url': 'https://news.google.com/rss/articles/CBMikwFBVV95cUxPd2ZnMnNwSWo2ZGhVSUJuNHd5S1Y3WUNWSkM4a0h5aHZQWU8tdzdlaW9pb25RUnI2cEwyZGtTemo5VUgwTDNHLVppNkw2MXdsbTRnb0UteHhtaHgxV043ZE9ZeG5aLUlCTzBGSHc1TFJzaHJsZENObzMxdTlvaEcyaG9vSjlRSTFWYXJEelF6RkRETnc?oc=5',
  'content': 'How DeepSeek is upending AI innovation and investment after sending tech leaders reeling - New York Post'},
 {'url': 'https://news.google.com/rss/articles/CBMivwFBVV95cUxNUGdjLVE5dFpLaVZOcFY1djBRQXBLeTNZalptNmstNXlWRkpvX1U2aTJ5cDNiS3RNT2JzeGI1SnlzTXIyS2dWcEdieDB4R1kxSEo2eXUydlRkVWlzOGdUTnVCQ2NwNjNjaFpCdVpxQkphZXYxLU9BaXhBWmdVYWVjQnY1N3Q1aUtqaER5LV9WVlNWZ3BXMk5WR0gwWnlIU3RIazJZelZJQUM1ek12ZDFodEg1eDFaRm56eTR5UEh3VQ?oc=5',
  'content': 'DeepSeek Marks The End Of The First Phase Of The AI Investment Boom - Forbes'}]

再用 tool 装饰器


```python
from langchain.tools import tool
from typing import List, Dict


# Create a tool for searching news by keyword
@tool
def search_keyword(query: str) -> List[Dict[str, str]]:
    """Look up news by keyword"""
    print(query)
    news_tool = GoogleNews()
    return news_tool.search_by_keyword(query, k=5)

	
# Execution Results
search_keyword.invoke({"query": "LangChain AI"})
```

# bind_tools

`bind_tools` 是 LangChain 中的一个强大功能，用于将自定义工具与大语言模型 (LLMs) 集成，从而实现增强的 AI 工作流。

接下来展示如何创建、绑定工具、解析并执行输出，并将它们集成到 `AgentExecutor` 中。

![](https://python.langchain.com/assets/images/tool_calling_components-bef9d2bcb9d3706c2fe58b57bf8ccb60.png)


```python
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


# Define the tools
@tool
def get_word_length(word: str) -> int:
    """Return the length of the given text"""
    return len(word)


@tool
def add_function(a: float, b: float) -> float:
    """Add two numbers together"""
    return a + b
	

tools = [get_word_length, add_function]
```

现在，让我们使用 `bind_tools` 函数将定义的工具与特定的大语言模型 (LLM) 关联起来。


```python
from langchain_openai import ChatOpenAI

# Create a model
llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-3B-Instruct',
	temperature=0.2,
)

# Tool binding
llm_with_tools = llm.bind_tools(tools)
```

这是绑定函数的说明


```python
import json

print(json.dumps(llm_with_tools.kwargs, indent=2))
```


```python
llm_with_tools.invoke(
    "What is the length of the given text 'LangChain OpenTutorial'?"
)
```

结果存储在 `tool_calls` 中。让我们打印 `tool_calls`。

[注意]

- `name` 表示工具的名称。
- `args` 包含传递给工具的参数。


```python
from pprint import pprint

# Execution result
ret = llm_with_tools.invoke(
    "What is the length of the given text 'LangChain OpenTutorial'?"
)

pprint(ret.__dict__)
print(20*'-')
pprint(ret.tool_calls)
```

## tool 输出解析 JsonOutputToolsParser

接下来，我们将把 `llm_with_tools` 与 `JsonOutputToolsParser` 连接起来，以解析 `tool_calls` 并查看结果。

- `type` 表示工具的类型。
- `args` 包含传递给工具的参数。


```python
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser

# Tool Binding + Tool Parser
chain = llm_with_tools | JsonOutputToolsParser(tools=tools)

# Execution Result
tool_call_results = chain.invoke(
    "What is the length of the given text 'LangChain OpenTutorial'?"
)
print(tool_call_results)
```

`execute_tool_calls` 函数识别合适的工具，传递相应的 `args`，然后执行该工具。


```python
def execute_tool_calls(tool_call_results):
    """
    Function to execute the tool call results.

    :param tool_call_results: List of the tool call results
    :param tools: List of available tools
    """

    # Iterate over the list of the tool call results
    for tool_call_result in tool_call_results:
        # Tool name (function name)
        tool_name = tool_call_result["type"]
        # Tool arguments
        tool_args = tool_call_result["args"]

        # Find the tool that matches the name and execute it
        # Use the next() function to find the first matching tool
        matching_tool = next((tool for tool in tools if tool.name == tool_name), None)
        if matching_tool:
            # Execute the tool
            result = matching_tool.invoke(tool_args)
            print(
                f"[Executed Tool] {tool_name} [Args] {tool_args}\n[Execution Result] {result}"
            )
        else:
            print(f"Warning: Unable to find the tool corresponding to {tool_name}.")


# Execute the tool calls
execute_tool_calls(tool_call_results)
```

## 将工具与解析器绑定以执行

这次，我们将工具绑定、解析结果和执行工具调用的整个过程合并为一个步骤。

- `llm_with_tools`：绑定工具的LLM模型。
- `JsonOutputToolsParser`：处理工具调用结果的解析器。
- `execute_tool_calls`：执行工具调用结果的函数。

[流程摘要]

1. 将工具绑定到模型。
2. 解析工具调用的结果。
3. 执行工具调用的结果。


```python
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser

# bind_tools + Parser + Execution
chain = llm_with_tools | JsonOutputToolsParser(tools=tools) | execute_tool_calls
```


```python
chain.invoke("What is the length of the given text 'LangChain OpenTutorial'?")
```


```python
# Execution Result 2
chain.invoke("114.5 + 121.2")

# Double check
print(114.5 + 121.2)
```

## 将工具与Agent和`AgentExecutor`绑定

`bind_tools` 提供可以由模型使用的工具（schemas）。

`AgentExecutor` 创建一个执行循环，用于执行诸如调用LLM、路由到合适的工具、执行工具以及重新调用模型等任务。


```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Create an Agent prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create a model
llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-3B-Instruct',
	temperature=0.2,
)

```


```python
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Use the tools defined previously
tools = [get_word_length, add_function]

# Create an Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)
```


```python
# Execute the Agent
result = agent_executor.invoke(
    {"input": "What is the length of the given text 'LangChain OpenTutorial'?"}
)

# Execution Result
print(result["output"])
```


```python
# Execute the Agent
result = agent_executor.invoke({"input": "Calculate the result of 114.5 + 121.2"})

# Execution Result
print(result["output"])
```

小模型的性能还是会存在解析和 reasoning 方面的错误

# Tool Calling Agent

在 LangChain 中，**工具调用**（tool calling）允许模型检测何时调用一个或多个**工具**，以及需要将什么输入传递给这些工具。

在进行 API 调用时，你可以定义工具，并智能地引导模型生成结构化对象，例如 JSON，这些对象包含调用这些工具所需的参数。

工具 API 的目标是提供比标准的文本生成或聊天 API 更加可靠的有效和有用的**工具调用**生成。

你可以创建代理（agents），这些代理会迭代地调用工具并接收结果，直到通过整合这些结构化输出解决查询，并将多个工具绑定到一个工具调用的聊天模型，让模型选择调用哪些工具。

这代表了一个更加**通用的版本**，它是为 OpenAI 的特定工具调用风格设计的 OpenAI 工具代理的扩展。

这个代理使用 LangChain 的 `ToolCall` 接口，支持比 OpenAI 更广泛的提供者实现，包括 `Anthropic`、`Google Gemini` 和 `Mistral` 等。

![](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/15-Agent/assets/15-agent-agent-concept.png?raw=1)

## 创建工具

LangChain 允许你定义自定义工具，供你的代理与之交互。你可以创建用于搜索新闻或执行 Python 代码的工具。

`@tool` 装饰器用于创建工具：
- `TavilySearchResults` 是一个用于搜索新闻的工具。
- `PythonREPL` 是一个用于执行 Python 代码的工具。


```python
from langchain.tools import tool
from typing import List, Dict, Annotated
from langchain_community.tools import TavilySearchResults
from langchain_experimental.utilities import PythonREPL


# Creating tool for searching news
@tool
def search_news(query: str) -> List[Dict[str, str]]:
    """Search news by input keyword using Tavily Search API"""
    news_tool = TavilySearchResults(
        max_results=3,
        include_answer=True,
        include_raw_content=True,
        include_images=True,
        # search_depth="advanced",
        # include_domains = [],
        # exclude_domains = []
    )
    return news_tool.invoke(query, k=3)


# Creating tool for executing python code
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this tool to execute Python code. If you want to see the output of a value,
    you should print it using print(...). This output is visible to the user."""
    result = ""
    try:
        result = PythonREPL().run(code)
    except BaseException as e:
        print(f"Failed to execute. Error: {repr(e)}")
    finally:
        return result


print(f"Tool name: {search_news.name}")
print(f"Tool description: {search_news.description}")
print(f"Tool args: {search_news.args}")

print('-'*20)
print(f"Tool name: {python_repl_tool.name}")
print(f"Tool description: {python_repl_tool.description}")
print(f"Tool args: {python_repl_tool.args}")

```

    Tool name: search_news
    Tool description: Search news by input keyword using Tavily Search API
    Tool args: {'query': {'title': 'Query', 'type': 'string'}}
    --------------------
    Tool name: python_repl_tool
    Tool description: Use this tool to execute Python code. If you want to see the output of a value,
        you should print it using print(...). This output is visible to the user.
    Tool args: {'code': {'description': 'The python code to execute to generate your chart.', 'title': 'Code', 'type': 'string'}}



```python
search_news('2024')
```




    [{'url': 'https://en.wikipedia.org/wiki/2024_in_the_United_States',
      'content': "In the Senate, at least six seats, those of Senators Tom Carper from Delaware, Mike Braun from Indiana, Ben Cardin from Maryland, Debbie Stabenow from Michigan, Mitt Romney from Utah, and Joe Manchin from West Virginia, will be open contests; the seat of the late Dianne Feinstein is also expected to be an open contest with Feinstein's immediate replacement, Laphonza Butler, expected to serve on an interim basis.[1][2][3]\nConcerning state governments, 11 states and two territories will hold gubernatorial elections, and most states and territories will hold elections for their legislatures. Contents\n2024 in the United States\nThe following is a list of predicted and scheduled events of the year 2024 in the United States, that have not yet occurred.\n With former president Donald Trump's declaration to run for the office again, the election may possibly be a rematch of the 2020 election, although the June 2023 indictment of Donald Trump may have a significant impact on Trump's presidential campaign. In the federal government, the offices of the president, vice president, all 435 seats of the House of Representatives, and roughly one third of the Senate. ←\n→\nElections[edit]\nThe US general elections will be held on November 5 of this year."},
     {'url': 'https://abcnews.go.com/Entertainment/abc-news-year-2024-back-years-major-news/story?id=116448091',
      'content': 'ABC News\' \'The Year: 2024\' looks back at this year\'s major news and entertainment events - ABC News ABC News ABC News\' \'The Year: 2024\' looks back at this year\'s major news and entertainment events As the world gears up for 2025, it leaves behind a year of war, political shifts, pop culture moments, sporting triumphs, lost stars and more. ABC News was there to chronicle every moment and will look back at this year\'s defining events in a two-hour special, "The Year: 2024," which airs Thursday, Dec. 26 at 9 p.m. ET, and streams afterwards on Hulu. The special also explores how the love lives of some of our favorite stars evolved this year. ABC News Live'},
     {'url': 'https://en.wikipedia.org/wiki/2024',
      'content': 'May 8 – In North Macedonian elections, the right-wing party VMRO-DPMNE wins in a landslide in the parliamentary elections, while its presidential candidate Gordana Siljanovska-Davkova is elected as the first female president of the country in the second round of the presidential election.[88][89] July 13 – While campaigning for the 2024 United States presidential election, former President Donald Trump is shot in the right ear in an assassination attempt at a rally he held near Butler, Pennsylvania.[139] July 28 – 2024 Venezuelan presidential election: Incumbent President Nicolás Maduro declares victory against opposition candidate Edmundo González Urrutia amid alleged irregularities, causing numerous South American states to refuse to acknowledge the results or suspend diplomatic relations with the Maduro government and sparking nationwide protests.[151]'}]




```python
# Creating tools
tools = [search_news, python_repl_tool]
```

## 构建代理提示

- `chat_history`：此变量存储对话历史记录，如果你的代理支持多轮对话，则使用此变量。（否则，可以省略此项。）
- `agent_scratchpad`：此变量作为临时存储，用于存放中间变量。
- `input`：此变量代表用户的输入。


```python
from langchain_core.prompts import ChatPromptTemplate

# Creating prompt
# Prompt is a text that describes the task the model should perform. (input the name and role of the tool)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `search_news` tool for searching keyword related news.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
```

## 创建代理

使用 `create_tool_calling_agent` 函数定义一个代理。


```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent

# Creating LLM
llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-3B-Instruct',
	temperature=0.2,
)

# Creating Agent
agent = create_tool_calling_agent(llm, tools, prompt)
```

## `AgentExecutor`

`AgentExecutor` 是一个用于管理使用工具的代理的类。

**关键属性**
- `agent`：负责创建计划并在执行循环的每个步骤中确定行动的底层代理。
- `tools`：包含代理被授权使用的所有有效工具的列表。
- `return_intermediate_steps`：布尔标志，决定是否返回代理在执行过程中所采取的中间步骤以及最终输出。
- `max_iterations`：代理在执行循环终止之前可以采取的最大步骤数。
- `max_execution_time`：执行循环允许运行的最长时间。
- `early_stopping_method`：定义当代理未返回 `AgentFinish` 时如何处理的方式。（"force" 或 "generate"）
  - `"force"`：返回一个字符串，表示执行循环由于达到时间或迭代限制而被停止。
  - `"generate"`：调用代理的 LLM 链一次，根据之前的步骤生成最终答案。
- `handle_parsing_errors`：指定如何处理解析错误。（可以设置为 `True`、`False`，或提供自定义错误处理函数。）
- `trim_intermediate_steps`：修剪中间步骤的方法。（可以设置为 `-1` 以保留所有步骤，或提供自定义修剪函数。）

**关键方法**
1. `invoke`：执行代理。
2. `stream`：流式传输达到最终输出所需的步骤。

**关键特性**
1. **工具验证**：确保工具与代理兼容。
2. **执行控制**：设置最大迭代次数和执行时间限制来管理代理行为。
3. **错误处理**：提供多种处理输出解析错误的选项。
4. **中间步骤管理**：允许修剪中间步骤或返回调试选项。
5. **异步支持**：支持异步执行和结果的流式传输。

**优化建议**
- 设置适当的 `max_iterations` 和 `max_execution_time` 值来管理执行时间。
- 使用 `trim_intermediate_steps` 来优化内存使用。
- 对于复杂任务，使用 `stream` 方法来逐步监控结果。


```python
from langchain.agents import AgentExecutor

# Create AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    max_execution_time=10,
    handle_parsing_errors=True,
)

# Run AgentExecutor
result = agent_executor.invoke({"input": "Search news about AI Agent in 2025."})

print("Agent execution result:")
print(result["output"])
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `search_news` with `{'query': 'AI Agent 2025'}`
    
    
    [0m[36;1m[1;3m[{'url': 'https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks', 'content': 'According to leading experts from Stanford Institute for Human-Centered AI, one major trend is the rise of collaborative AI systems where multiple specialized agents work together, with humans providing high-level guidance. I expect to see more focus on multimodal AI models in education, including in processing speech and images. AI Agents Work Together In 2025, we will see a significant shift from relying on individual AI models to using systems where multiple AI agents of diverse expertise work together. As an example, we recently introduced the\xa0Virtual Lab, where a professor AI agent leads a team of AI scientist agents (e.g., AI chemist, AI biologist) to tackle challenging, open-ended research, with a human researcher providing high-level feedback. We will experience an emerging paradigm of research around how humans work together with AI agents.'}, {'url': 'https://www.forbes.com/sites/lutzfinger/2025/01/05/ai-agents-in-2025-what-enterprise-leaders-need-to-know/', 'content': 'AI Agents In 2025: What Enterprise Leaders Need To Know AI Agents In 2025: What Enterprise Leaders Need To Know AI Agents for the Enterprise will be the focus of 2025 To see what AI agents can do in 2025, let’s consider a simple example: an email-answering tool. Let’s improve our tool by building AI agents within a workflow. The Workflow of AI Agents: More Than Generative AI AI models can be connected or "chained" to build workflows where the output of one model becomes the input for the next. AI Agent Workflows: Input - Orchestration - Control - Actions - Synthesizing 2025 - AI Agents for the Enterprise Follow me here on Forbes or on LinkedIn for more of my 2025 AI predictions.'}, {'url': 'https://www.godofprompt.ai/blog/ai-agents-you-cant-miss', 'content': 'Explore 10+ AI agents that are reshaping industries in 2025. From ChatGPT to DeepSeek-R1, discover how AI is becoming more intelligent, efficient, and essential for businesses and individuals alike.'}][0m[32;1m[1;3mHere are some relevant news articles about AI Agents in 2025:
    
    1. [According to leading experts from Stanford Institute for Human-Centered AI, one major trend is the rise of collaborative AI systems where multiple specialized agents work together, with humans providing high-level guidance.](https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks)
    
    2. [AI Agents In 2025: What Enterprise Leaders Need To Know](https://www.forbes.com/sites/lutzfinger/2025/01/05/ai-agents-in-2025-what-enterprise-leaders-need-to-know/) - This article discusses AI Agents for the Enterprise, focusing on AI models being connected or "chained" to build workflows where the output of one model becomes the input for the next. It also mentions AI Agent Workflows: Input - Orchestration - Control - Actions - Synthesizing.
    
    3. [Explore 10+ AI agents that are reshaping industries in 2025. From ChatGPT to DeepSeek-R1, discover how AI is becoming more intelligent, efficient, and essential for businesses and individuals alike.](https://www.godofprompt.ai/blog/ai-agents-you-cant-miss)
    
    These articles provide insights into the future of AI Agents, including their collaborative nature, their potential impact on enterprises, and the development of AI Agent Workflows.[0m
    
    [1m> Finished chain.[0m
    Agent execution result:
    Here are some relevant news articles about AI Agents in 2025:
    
    1. [According to leading experts from Stanford Institute for Human-Centered AI, one major trend is the rise of collaborative AI systems where multiple specialized agents work together, with humans providing high-level guidance.](https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks)
    
    2. [AI Agents In 2025: What Enterprise Leaders Need To Know](https://www.forbes.com/sites/lutzfinger/2025/01/05/ai-agents-in-2025-what-enterprise-leaders-need-to-know/) - This article discusses AI Agents for the Enterprise, focusing on AI models being connected or "chained" to build workflows where the output of one model becomes the input for the next. It also mentions AI Agent Workflows: Input - Orchestration - Control - Actions - Synthesizing.
    
    3. [Explore 10+ AI agents that are reshaping industries in 2025. From ChatGPT to DeepSeek-R1, discover how AI is becoming more intelligent, efficient, and essential for businesses and individuals alike.](https://www.godofprompt.ai/blog/ai-agents-you-cant-miss)
    
    These articles provide insights into the future of AI Agents, including their collaborative nature, their potential impact on enterprises, and the development of AI Agent Workflows.


## 使用 Stream 输出检查逐步结果

我们将使用 `AgentExecutor` 的 `stream()` 方法来流式传输代理的中间步骤。

`stream()` 的输出在 (Action, Observation) 对之间交替，最终如果目标达成，将以代理的答案结束。

流程如下所示：

1. Action 输出
2. Observation 输出
3. Action 输出
4. Observation 输出

...（继续直到目标达成）...

然后，代理将在目标达成后得出最终答案。

以下表格总结了你将在输出中遇到的内容：

| 输出 | 描述 |
|------|------|
| Action | `actions`：表示 `AgentAction` 或其子类。<br>`messages`：与动作调用对应的聊天消息。 |
| Observation | `steps`：记录代理的工作，包括当前的动作和其观察结果。<br>`messages`：包含函数调用结果（即观察结果）的聊天消息。 |
| Final Answer | `output`：表示 `AgentFinish` 信号。<br>`messages`：包含最终输出的聊天消息。 |


```python
from langchain.agents import AgentExecutor

# Create AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
)

# Run in streaming mode
result = agent_executor.stream({"input": "Search news about AI Agent in 2025."})

for step in result:
    # Print intermediate steps
    print(step)
    print("===" * 20)
```

    {'actions': [ToolAgentAction(tool='search_news', tool_input={'query': 'AI Agent 2025'}, log="\nInvoking: `search_news` with `{'query': 'AI Agent 2025'}`\n\n\n", message_log=[AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'chatcmpl-tool-cf8525019f5847519566061e0e6647c6', 'function': {'arguments': '{"query": "AI Agent 2025"}', 'name': 'search_news'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'Qwen2.5-3B-Instruct'}, id='run-a877dfea-5a20-4970-96da-0f3483298f7e', tool_calls=[{'name': 'search_news', 'args': {'query': 'AI Agent 2025'}, 'id': 'chatcmpl-tool-cf8525019f5847519566061e0e6647c6', 'type': 'tool_call'}], tool_call_chunks=[{'name': 'search_news', 'args': '{"query": "AI Agent 2025"}', 'id': 'chatcmpl-tool-cf8525019f5847519566061e0e6647c6', 'index': 0, 'type': 'tool_call_chunk'}])], tool_call_id='chatcmpl-tool-cf8525019f5847519566061e0e6647c6')], 'messages': [AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'chatcmpl-tool-cf8525019f5847519566061e0e6647c6', 'function': {'arguments': '{"query": "AI Agent 2025"}', 'name': 'search_news'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'Qwen2.5-3B-Instruct'}, id='run-a877dfea-5a20-4970-96da-0f3483298f7e', tool_calls=[{'name': 'search_news', 'args': {'query': 'AI Agent 2025'}, 'id': 'chatcmpl-tool-cf8525019f5847519566061e0e6647c6', 'type': 'tool_call'}], tool_call_chunks=[{'name': 'search_news', 'args': '{"query": "AI Agent 2025"}', 'id': 'chatcmpl-tool-cf8525019f5847519566061e0e6647c6', 'index': 0, 'type': 'tool_call_chunk'}])]}
    ============================================================
    {'steps': [AgentStep(action=ToolAgentAction(tool='search_news', tool_input={'query': 'AI Agent 2025'}, log="\nInvoking: `search_news` with `{'query': 'AI Agent 2025'}`\n\n\n", message_log=[AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'chatcmpl-tool-cf8525019f5847519566061e0e6647c6', 'function': {'arguments': '{"query": "AI Agent 2025"}', 'name': 'search_news'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'Qwen2.5-3B-Instruct'}, id='run-a877dfea-5a20-4970-96da-0f3483298f7e', tool_calls=[{'name': 'search_news', 'args': {'query': 'AI Agent 2025'}, 'id': 'chatcmpl-tool-cf8525019f5847519566061e0e6647c6', 'type': 'tool_call'}], tool_call_chunks=[{'name': 'search_news', 'args': '{"query": "AI Agent 2025"}', 'id': 'chatcmpl-tool-cf8525019f5847519566061e0e6647c6', 'index': 0, 'type': 'tool_call_chunk'}])], tool_call_id='chatcmpl-tool-cf8525019f5847519566061e0e6647c6'), observation=[{'url': 'https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks', 'content': 'According to leading experts from Stanford Institute for Human-Centered AI, one major trend is the rise of collaborative AI systems where multiple specialized agents work together, with humans providing high-level guidance. I expect to see more focus on multimodal AI models in education, including in processing speech and images. AI Agents Work Together In 2025, we will see a significant shift from relying on individual AI models to using systems where multiple AI agents of diverse expertise work together. As an example, we recently introduced the\xa0Virtual Lab, where a professor AI agent leads a team of AI scientist agents (e.g., AI chemist, AI biologist) to tackle challenging, open-ended research, with a human researcher providing high-level feedback. We will experience an emerging paradigm of research around how humans work together with AI agents.'}, {'url': 'https://www.forbes.com/sites/lutzfinger/2025/01/05/ai-agents-in-2025-what-enterprise-leaders-need-to-know/', 'content': 'AI Agents In 2025: What Enterprise Leaders Need To Know AI Agents In 2025: What Enterprise Leaders Need To Know AI Agents for the Enterprise will be the focus of 2025 To see what AI agents can do in 2025, let’s consider a simple example: an email-answering tool. Let’s improve our tool by building AI agents within a workflow. The Workflow of AI Agents: More Than Generative AI AI models can be connected or "chained" to build workflows where the output of one model becomes the input for the next. AI Agent Workflows: Input - Orchestration - Control - Actions - Synthesizing 2025 - AI Agents for the Enterprise Follow me here on Forbes or on LinkedIn for more of my 2025 AI predictions.'}, {'url': 'https://www.godofprompt.ai/blog/ai-agents-you-cant-miss', 'content': 'Explore 10+ AI agents that are reshaping industries in 2025. From ChatGPT to DeepSeek-R1, discover how AI is becoming more intelligent, efficient, and essential for businesses and individuals alike.'}])], 'messages': [FunctionMessage(content='[{"url": "https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks", "content": "According to leading experts from Stanford Institute for Human-Centered AI, one major trend is the rise of collaborative AI systems where multiple specialized agents work together, with humans providing high-level guidance. I expect to see more focus on multimodal AI models in education, including in processing speech and images. AI Agents Work Together In 2025, we will see a significant shift from relying on individual AI models to using systems where multiple AI agents of diverse expertise work together. As an example, we recently introduced the\xa0Virtual Lab, where a professor AI agent leads a team of AI scientist agents (e.g., AI chemist, AI biologist) to tackle challenging, open-ended research, with a human researcher providing high-level feedback. We will experience an emerging paradigm of research around how humans work together with AI agents."}, {"url": "https://www.forbes.com/sites/lutzfinger/2025/01/05/ai-agents-in-2025-what-enterprise-leaders-need-to-know/", "content": "AI Agents In 2025: What Enterprise Leaders Need To Know AI Agents In 2025: What Enterprise Leaders Need To Know AI Agents for the Enterprise will be the focus of 2025 To see what AI agents can do in 2025, let’s consider a simple example: an email-answering tool. Let’s improve our tool by building AI agents within a workflow. The Workflow of AI Agents: More Than Generative AI AI models can be connected or \\"chained\\" to build workflows where the output of one model becomes the input for the next. AI Agent Workflows: Input - Orchestration - Control - Actions - Synthesizing 2025 - AI Agents for the Enterprise Follow me here on Forbes or on LinkedIn for more of my 2025 AI predictions."}, {"url": "https://www.godofprompt.ai/blog/ai-agents-you-cant-miss", "content": "Explore 10+ AI agents that are reshaping industries in 2025. From ChatGPT to DeepSeek-R1, discover how AI is becoming more intelligent, efficient, and essential for businesses and individuals alike."}]', additional_kwargs={}, response_metadata={}, name='search_news')]}
    ============================================================
    {'output': 'Here are some relevant news articles about AI Agents in 2025:\n\n1. [According to leading experts from Stanford Institute for Human-Centered AI, one major trend is the rise of collaborative AI systems where multiple specialized agents work together, with humans providing high-level guidance.](https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks)\n\n2. [AI Agents In 2025: What Enterprise Leaders Need To Know](https://www.forbes.com/sites/lutzfinger/2025/01/05/ai-agents-in-2025-what-enterprise-leaders-need-to-know/) - This article discusses AI Agents for the Enterprise, focusing on how AI models can be connected or "chained" to build workflows where the output of one model becomes the input for the next.\n\n3. [Explore 10+ AI agents that are reshaping industries in 2025. From ChatGPT to DeepSeek-R1, discover how AI is becoming more intelligent, efficient, and essential for businesses and individuals alike.](https://www.godofprompt.ai/blog/ai-agents-you-cant-miss)\n\nThese articles provide insights into the expected trends and developments in AI Agents for both research and enterprise applications in the year 2025.', 'messages': [AIMessage(content='Here are some relevant news articles about AI Agents in 2025:\n\n1. [According to leading experts from Stanford Institute for Human-Centered AI, one major trend is the rise of collaborative AI systems where multiple specialized agents work together, with humans providing high-level guidance.](https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks)\n\n2. [AI Agents In 2025: What Enterprise Leaders Need To Know](https://www.forbes.com/sites/lutzfinger/2025/01/05/ai-agents-in-2025-what-enterprise-leaders-need-to-know/) - This article discusses AI Agents for the Enterprise, focusing on how AI models can be connected or "chained" to build workflows where the output of one model becomes the input for the next.\n\n3. [Explore 10+ AI agents that are reshaping industries in 2025. From ChatGPT to DeepSeek-R1, discover how AI is becoming more intelligent, efficient, and essential for businesses and individuals alike.](https://www.godofprompt.ai/blog/ai-agents-you-cant-miss)\n\nThese articles provide insights into the expected trends and developments in AI Agents for both research and enterprise applications in the year 2025.', additional_kwargs={}, response_metadata={})]}
    ============================================================


## 使用用户定义的函数自定义中间步骤输出

你可以定义以下 3 个函数来自定义中间步骤的输出：

- `tool_callback`：此函数处理工具调用生成的输出。
- `observation_callback`：此函数处理观察数据输出。
- `result_callback`：此函数允许你处理最终答案的输出。


```python
from typing import Dict, Any


# Create AgentStreamParser class
class AgentStreamParser:
    def __init__(self):
        pass

    def tool_callback(self, tool: Dict[str, Any]) -> None:
        print("\n=== Tool Called ===")
        print(f"Tool: {tool.get('tool')}")
        print(f"Input: {tool.get('tool_input')}")
        print("==================\n")

    def observation_callback(self, step: Dict[str, Any]) -> None:
        print("\n=== Observation ===")
        observation_data = step["steps"][0].observation
        print(f"Observation: {observation_data}")
        print("===================\n")

    def result_callback(self, result: str) -> None:
        print("\n=== Final Answer ===")
        print(result)
        print("====================\n")

    def process_agent_steps(self, step: Dict[str, Any]) -> None:
        if "actions" in step:
            for action in step["actions"]:
                self.tool_callback(
                    {"tool": action.tool, "tool_input": action.tool_input}
                )
        elif "output" in step:
            self.result_callback(step["output"])
        else:
            self.observation_callback(step)


# Create AgentStreamParser instance
agent_stream_parser = AgentStreamParser()
```


```python
# Run in streaming mode
result = agent_executor.stream({"input": "Generate a array from 0 to 1 with the stride of 0.1 using numpy."})
# result = agent_executor.stream({"input": "Search news about AI Agent in 2025."})


for step in result:
    agent_stream_parser.process_agent_steps(step)
```

    
    === Tool Called ===
    Tool: numpy_array_generator
    Input: {'start': 1, 'step': 0.1}
    ==================
    
    
    === Observation ===
    Observation: numpy_array_generator is not a valid tool, try one of [search_news, python_repl_tool].
    ===================
    
    
    === Tool Called ===
    Tool: python_repl_tool
    Input: {'code': 'import numpy as np\nnp.arange(0, 1, 0.1)'}
    ==================
    
    
    === Observation ===
    Observation: 
    ===================
    
    
    === Tool Called ===
    Tool: python_repl_tool
    Input: {'code': 'import numpy as np\nnp.arange(0, 1, 0.1)'}
    ==================
    
    
    === Observation ===
    Observation: 
    ===================
    
    
    === Tool Called ===
    Tool: python_repl_tool
    Input: {'code': 'import numpy as np\nnp.arange(0, 1, 0.1)'}
    ==================
    
    
    === Observation ===
    Observation: 
    ===================
    
    
    === Final Answer ===
    The numpy array from 0 to 1 with a stride of 0.1 has been successfully generated. Here it is:
    
    ```
    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ```
    
    Is there anything else I can assist you with?
    ====================
    


## 与之前的对话历史进行代理通信

为了记住过去的对话，你可以将 `AgentExecutor` 包装在 `RunnableWithMessageHistory` 中。

有关 `RunnableWithMessageHistory` 的更多细节，请参阅以下链接。

**参考**
- [LangChain Python API Reference > langchain: 0.3.14 > core > runnables > langchain_core.runnables.history > RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html)


```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Create a dictionary to store session_id
store = {}


# Function to get session history based on session_id
def get_session_history(session_ids):
    if session_ids not in store:  # If session_id is not in store
        # Create a new ChatMessageHistory object and store it in store
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # Return session history for the corresponding session_id


# Create an agent with chat message history
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # Chat session_id
    get_session_history,
    # The key for the question input in the prompt: "input"
    input_messages_key="input",
    # The key for the message input in the prompt: "chat_history"
    history_messages_key="chat_history",
)
```


```python
# Request streaming output for the query
response = agent_with_chat_history.stream(
    {"input": "Hello! My name is Teddy!"},
    # Set session_id
    config={"configurable": {"session_id": "abc123"}},
)

# Check the output
for step in response:
    agent_stream_parser.process_agent_steps(step)
```

    
    === Final Answer ===
    Hello Teddy! It's nice to meet you. How can I assist you today? Do you have any specific questions or topics you'd like to explore?
    ====================
    



```python
# Request streaming output for the query
response = agent_with_chat_history.stream(
    {"input": "What is my name?"},
    # Set session_id
    config={"configurable": {"session_id": "abc123"}},
)

# Check the output
for step in response:
    agent_stream_parser.process_agent_steps(step)
```

    
    === Final Answer ===
    It seems like you've already provided your name as Teddy. If you have any other questions or need information about something else, feel free to ask!
    ====================
    



```python
# Request streaming output for the query
response = agent_with_chat_history.stream(
    {
        "input": "My email address is teddy@teddynote.com. The company name is TeddyNote Co., Ltd."
    },
    # Set session_id
    config={"configurable": {"session_id": "abc123"}},
)

# Check the output
for step in response:
    agent_stream_parser.process_agent_steps(step)
```

    
    === Tool Called ===
    Tool: search_news
    Input: {'query': 'TeddyNote Co., Ltd.'}
    ==================
    
    
    === Observation ===
    Observation: [{'url': 'https://www.youtube.com/@teddynote', 'content': '데이터 분석, 머신러닝, 딥러닝, LLM 에 대한 내용을 다룹니다. 연구보다는 개발에 관심이 많습니다 🙇\u200d♂️🔥 "테디노트의 RAG 비법노트" 랭체인'}, {'url': 'https://github.com/teddynote', 'content': 'By company size. Enterprises Small and medium teams Startups Nonprofits By use case. DevSecOps DevOps CI/CD View all use cases By industry ... teddynote.github.io teddynote.github.io Public. Forked from mmistakes/minimal-mistakes. 📐 Jekyll theme for building a personal site, blog, project documentation, or portfolio.'}, {'url': 'https://github.com/teddylee777', 'content': 'Jupyter Notebook\n1\n4\nConv2d and MaxPool2d Calculator for PyTorch\nPython\n18\n1\nStreamlit 튜토리얼 😁\nJupyter Notebook\n13\n12\n주가 종목 패턴 발굴기\nJupyter Notebook\n14\n12\n586\ncontributions\nin the last year\nContribution activity\nJanuary 2024\nSeeing something unexpected? Teddy Lee\nteddylee777\nAchievements\nAchievements\nHighlights\nBlock or report teddylee777\nPrevent this user from interacting with your repositories and sending you notifications.\n Jupyter Notebook\n58\n16\nForked from lovedlim/tensorflow\n텐서플로 도서 예제 파일입니다.\n Samsung Electronics\n테디노트 Blog\n테디노트 YouTube\n@teddynote\nLinkedIn\n💻 (This repository is intented for helping whom are interested in machine learning study)\nJupyter Notebook\n2.3k\n789\n머신러닝/딥러닝(PyTorch, TensorFlow) 전용 도커입니다.'}]
    ===================
    
    
    === Final Answer ===
    Here are some recent news and information related to your company, TeddyNote Co., Ltd.:
    
    1. **TeddyNote Co., Ltd. on YouTube**: They have a YouTube channel where they discuss topics related to data analysis, machine learning, deep learning, and Large Language Models (LLM). They seem to have a focus on development rather than research. You can check out their channel [here](https://www.youtube.com/@teddynote).
    
    2. **TeddyNote Co., Ltd. on GitHub**: 
       - **Company Size**: They cater to small and medium-sized teams, startups, and nonprofits.
       - **Use Cases**: They support DevSecOps and DevOps, CI/CD.
       - **Public Repository**: You can view their public repository [here](https://github.com/teddynote).
       - **Personal Site**: They have a personal site and blog available at [teddynote.github.io](https://teddynote.github.io/).
       - **Contributions**: They have contributed to various projects, including a Jupyter Notebook for a Conv2d and MaxPool2d Calculator for PyTorch, a Streamlit tutorial, and a stock price pattern analysis tool. You can see their contributions [here](https://github.com/teddylee777).
    
    3. **TeddyLee777 on GitHub**: This is likely a personal GitHub account associated with TeddyNote Co., Ltd. They have contributed to various projects, including a TensorFlow book example repository and a Docker image for machine learning study.
    
    If you need more detailed information or have any specific questions about these resources, feel free to ask!
    ====================
    



```python
# Request streaming output for the query
response = agent_with_chat_history.stream(
    {
        "input": "Search the latest news and write it as the body of the email. "
        "The recipient is `Ms. Sally` and the sender is my personal information."
        "Write in a polite tone, and include appropriate greetings and closings at the beginning and end of the email."
    },
    # Set session_id
    config={"configurable": {"session_id": "abc123"}},
)

# Check the output
for step in response:
    agent_stream_parser.process_agent_steps(step)
```

    
    === Tool Called ===
    Tool: search_news
    Input: {'query': 'TeddyNote Co., Ltd latest news'}
    ==================
    
    
    === Tool Called ===
    Tool: python_repl_tool
    Input: {'code': 'import email; from email.mime.multipart import MIMEMultipart; from email.mime.text import MIMEText; msg = MIMEMultipart(); msg[\'From\'] = \'teddy@teddynote.com\'; msg[\'To\'] = \'sally@example.com\'; msg[\'Subject\'] = \'Latest News from TeddyNote Co., Ltd\'; body = """Here are the latest news and updates from TeddyNote Co., Ltd.:\n\n1. **TeddyNote Co., Ltd. on YouTube**: They have a YouTube channel where they discuss topics related to data analysis, machine learning, deep learning, and Large Language Models (LLM). They seem to have a focus on development rather than research. You can check out their channel [here](https://www.youtube.com/@teddynote).\n\n2. **TeddyNote Co., Ltd. on GitHub**: \n   - **Company Size**: They cater to small and medium-sized teams, startups, and nonprofits.\n   - **Use Cases**: They support DevSecOps and DevOps, CI/CD.\n   - **Public Repository**: You can view their public repository [here](https://github.com/teddynote).\n   - **Personal Site**: They have a personal site and blog available at [teddynote.github.io](https://teddynote.github.io/).\n   - **Contributions**: They have contributed to various projects, including a Jupyter Notebook for a Conv2d and MaxPool2d Calculator for PyTorch, a Streamlit tutorial, and a stock price pattern analysis tool. You can see their contributions [here](https://github.com/teddylee777).\n\n3. **TeddyLee777 on GitHub**: This is likely a personal GitHub account associated with TeddyNote Co., Ltd. They have contributed to various projects, including a TensorFlow book example repository and a Docker image for machine learning study.\n\nIf you need more detailed information or have any specific questions about these resources, feel free to ask!"""; msg.attach(MIMEText(body, \'plain\')); return msg.as_string()'}
    ==================
    
    
    === Tool Called ===
    Tool: send_email
    Input: {'to': 'sally@example.com', 'subject': 'Latest News from TeddyNote Co., Ltd', 'body': '...', 'sender': 'teddy@teddynote.com'}
    ==================
    
    
    === Tool Called ===
    Tool: email_status
    Input: {'email_id': '...'}
    ==================
    
    
    === Observation ===
    Observation: [{'url': 'https://www.threads.net/@teddynote', 'content': '60 Followers • 44 Threads • 데이터 & AI. See the latest conversations with @teddynote.'}, {'url': 'https://github.com/teddylee777', 'content': 'Jupyter Notebook\n1\n4\nConv2d and MaxPool2d Calculator for PyTorch\nPython\n18\n1\nStreamlit 튜토리얼 😁\nJupyter Notebook\n13\n12\n주가 종목 패턴 발굴기\nJupyter Notebook\n14\n12\n586\ncontributions\nin the last year\nContribution activity\nJanuary 2024\nSeeing something unexpected? Teddy Lee\nteddylee777\nAchievements\nAchievements\nHighlights\nBlock or report teddylee777\nPrevent this user from interacting with your repositories and sending you notifications.\n Jupyter Notebook\n58\n16\nForked from lovedlim/tensorflow\n텐서플로 도서 예제 파일입니다.\n Samsung Electronics\n테디노트 Blog\n테디노트 YouTube\n@teddynote\nLinkedIn\n💻 (This repository is intented for helping whom are interested in machine learning study)\nJupyter Notebook\n2.3k\n789\n머신러닝/딥러닝(PyTorch, TensorFlow) 전용 도커입니다.'}, {'url': 'https://langchain-opentutorial.gitbook.io/langchain-opentutorial/15-agent/03-agent', 'content': 'Best regards, Teddy teddy@teddynote.com TeddyNote Co., Ltd. --- Feel free to modify any part of the email as you see fit! >>>>>'}]
    ===================
    
    
    === Observation ===
    Observation: SyntaxError("'return' outside function", ('<string>', 14, 152, None, 14, 174))
    ===================
    
    
    === Observation ===
    Observation: send_email is not a valid tool, try one of [search_news, python_repl_tool].
    ===================
    
    
    === Observation ===
    Observation: email_status is not a valid tool, try one of [search_news, python_repl_tool].
    ===================
    
    
    === Final Answer ===
    It seems there was an issue with the previous steps. Let's proceed with creating the email body using the news and information we gathered. Here is the body of the email:
    
    ---
    
    Hello Ms. Sally,
    
    I hope this email finds you well. I wanted to share some recent news and updates from TeddyNote Co., Ltd.:
    
    1. **TeddyNote Co., Ltd. on YouTube**: They have a YouTube channel where they discuss topics related to data analysis, machine learning, deep learning, and Large Language Models (LLM). They seem to have a focus on development rather than research. You can check out their channel [here](https://www.youtube.com/@teddynote).
    
    2. **TeddyNote Co., Ltd. on GitHub**:
       - **Company Size**: They cater to small and medium-sized teams, startups, and nonprofits.
       - **Use Cases**: They support DevSecOps and DevOps, CI/CD.
       - **Public Repository**: You can view their public repository [here](https://github.com/teddynote).
       - **Personal Site**: They have a personal site and blog available at [teddynote.github.io](https://teddynote.github.io/).
       - **Contributions**: They have contributed to various projects, including a Jupyter Notebook for a Conv2d and MaxPool2d Calculator for PyTorch, a Streamlit tutorial, and a stock price pattern analysis tool. You can see their contributions [here](https://github.com/teddylee777).
    
    3. **TeddyLee777 on GitHub**: This is likely a personal GitHub account associated with TeddyNote Co., Ltd. They have contributed to various projects, including a TensorFlow book example repository and a Docker image for machine learning study.
    
    If you need more detailed information or have any specific questions about these resources, feel free to ask!
    
    Best regards,
    Teddy
    teddy@teddynote.com
    TeddyNote Co., Ltd.
    
    ---
    
    Please let me know if you need any further assistance or if there are any specific details you would like to include.
    ====================
    


# Agentic RAG

**Agentic RAG** 扩展了传统的 RAG（检索增强生成）系统，通过结合基于代理的方法，实现更复杂的信息检索和响应生成。该系统不仅仅局限于简单的文档检索和响应生成，还允许代理利用各种工具进行更智能的信息处理。这些工具包括用于访问最新信息的 `Tavily Search`、执行 Python 代码的能力以及自定义功能实现，所有这些都集成在 `LangChain` 框架中，为信息处理和生成任务提供全面的解决方案。

本教程演示了如何构建一个文档检索系统，使用 `FAISS DB` 来有效地处理和搜索 PDF 文档。以软件政策研究所的 AI Brief 为示例文档，我们将探索如何将基于 Web 的文档加载器、文本拆分器、向量存储和 `OpenAI` 嵌入结合起来，创建一个实际的 **Agentic RAG** 系统。该实现展示了如何将 `Retriever` 工具与各种 `LangChain` 组件有效结合，创建一个强大的文档搜索和响应生成管道。

## 创建工具


```python
from langchain_community.tools.tavily_search import TavilySearchResults

# Create a search tool instance that returns up to 6 results
search = TavilySearchResults(k=6)
```


```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.tools.retriever import create_retriever_tool

# Load and process the PDF
loader = PyPDFLoader("data/What-is-AI.pdf")

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


# Split the document
split_docs = loader.load_and_split(text_splitter)

# Create vector store
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
)
vector = FAISS.from_documents(split_docs, embeddings)

# Create retriever
retriever = vector.as_retriever()

# Create retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",
    description="use this tool to search information from the PDF document",
)
```

## 创建代理


```python
tools = [search, retriever_tool]
```


```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Initialize LLM
llm = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-3B-Instruct',
	temperature=0.2,
)

# Define prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `pdf_search` tool for searching information from the PDF document. "
            "If you can't find the information from the PDF document, use the `search` tool for searching information from the web.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
```

## 对话历史


```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Create a store for session histories
store = {}


def get_session_history(session_ids):
    if session_ids not in store:
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]


# Create agent with chat history
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
```


```python
def process_response(response):
    """
    Process and display streaming response from the agent.

    Args:
        response: Agent's streaming response iterator
    """
    for chunk in response:
        if chunk.get("output"):
            print(chunk["output"])
        elif chunk.get("actions"):
            for action in chunk["actions"]:
                print(f"\nTool Used: {action.tool}")
                print(f"Tool Input: {action.tool_input}")
                if action.log:
                    print(f"Tool Log: {action.log}")
```


```python
# Example 1: Searching in PDF
response = agent_with_chat_history.stream(
    {
        "input": "What information can you find about Samsung's AI model in the document?"
    },
    config={"configurable": {"session_id": "tutorial_session_1"}},
)
process_response(response)
```

    
    Tool Used: pdf_search
    Tool Input: {'query': 'Samsung AI model'}
    Tool Log: 
    Invoking: `pdf_search` with `{'query': 'Samsung AI model'}`
    
    
    
    
    Tool Used: search
    Tool Input: {'query': 'Samsung AI model'}
    Tool Log: 
    Invoking: `search` with `{'query': 'Samsung AI model'}`
    responded: The provided text does not contain specific information about Samsung's AI model. It seems to be a general introduction to AI, its components, and some real-world applications. To find information about Samsung's AI model, we might need to look for more detailed or specific documents or articles. Let me try searching the web for more relevant information.
    
    
    
    
    Tool Used: tavily_search_results_json
    Tool Input: {'query': 'Samsung AI model'}
    Tool Log: 
    Invoking: `tavily_search_results_json` with `{'query': 'Samsung AI model'}`
    responded: It appears that the 'search' tool is not available. Let's try using the 'tavily_search_results_json' tool to search the web for information about Samsung's AI model.
    
    
    
    
    The search results provide information about Samsung's AI model, specifically Samsung Gauss 2, which is described as a new GenAI model that improves Galaxy AI performance and efficiency. Here are some key points from the search results:
    
    1. **Samsung Gauss 2**: This model supports 9 to 14 human languages and several programming languages. Samsung claims that Balanced and Supreme models match or beat other AI models on tasks.
    
    2. **Galaxy S25 Series**: The Galaxy S25 series features advanced, efficient AI image processing with ProScaler11, achieving a 40% improvement in display image scaling quality. It also incorporates custom technology with Samsung’s mobile Digital Natural Image engine (mDNIe) embedded within the processor using Galaxy IP to enable greater display power efficiency.
    
    3. **Galaxy AI**: Samsung's Galaxy AI is described as a set of generative AI tools that brings features like live translation, generative photo editing, and more. The AI features are available on newer Samsung phones, but Samsung is making efforts to support these features on older models as well.
    
    4. **Samsung Gauss 2 on Device**: Samsung Gauss 2 is an on-device AI model, which means it processes data locally on the device rather than sending it to a cloud server.
    
    These results suggest that Samsung Gauss 2 is a significant advancement in their AI capabilities, particularly in improving Galaxy AI performance and efficiency. If you need more detailed information, you might want to look into the specific features and capabilities of Samsung Gauss 2 in more detail.



```python
# Example 1: Searching in PDF
response = agent_with_chat_history.stream(
    {
        "input": "List the devices using ai in your previous responese."
    },
    config={"configurable": {"session_id": "tutorial_session_1"}},
)
process_response(response)
```

    
    Tool Used: pdf_search
    Tool Input: {'query': 'devices using ai'}
    Tool Log: 
    Invoking: `pdf_search` with `{'query': 'devices using ai'}`
    
    
    
    Based on the information provided in the document, the devices mentioned that use AI are:
    
    1. **Smartphones**: The document mentions that AI is available on newer Samsung phones, indicating that smartphones are one of the devices using AI.
    2. **Galaxy S25 Series**: The document describes the Galaxy S25 series as featuring advanced, efficient AI image processing, which implies that this device uses AI.
    3. **Galaxy AI**: The document states that Galaxy AI is a set of generative AI tools available on newer Samsung phones, suggesting that the Galaxy S25 series and possibly other newer Samsung devices use AI.
    4. **Smart City Initiative**: The document provides an example of a government initiative using AI for real-time video analytics, advanced forensic investigation capabilities, and comprehensive operational intelligence. While it doesn't specify which devices are used, it implies that AI is used across various devices in this context.
    
    Therefore, the devices using AI mentioned in the document are:
    - Smartphones (Samsung)
    - Galaxy S25 Series
    - Galaxy AI (presumably available on newer Samsung phones)


# VLLM 部署本地大模型

```Bash
export CUDA_VISIBLE_DEVICES=5

modelpath=../DataCollection/officials/Qwen2.5-7B-Instruct
modelname=Qwen2.5-7B-Instruct

nohup python -m vllm.entrypoints.openai.api_server \
    --model $modelpath \
    --served-model-name $modelname \
    --port 5551 \
    --gpu-memory-utilization 0.4 \
    --dtype=half \
    > output.log 2>&1 &

```

### **基本模型选项**  

以下是 API 的基本选项：  

- **`model_name`** : `str`  
  该选项允许您选择适用的模型，也可以使用 `model` 作为别名。  

- **`temperature`** : `float` = 0.7  
  该选项用于设置采样温度（`temperature`）。取值范围为 **0 到 2**，较高的值（如 `0.8`）会使输出更加随机，而较低的值（如 `0.2`）会使输出更具**集中性和确定性**。  

- **`max_tokens`** : `int` | `None` = `None`  
  指定聊天补全（chat completion）中要生成的**最大 token 数**。该选项控制模型在一次调用中可以生成的文本长度。


```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
	base_url='http://localhost:5551/v1',
	api_key='EMPTY',
	model_name='Qwen2.5-7B-Instruct',
	temperature=0.2,
)

query = "Tell me one joke about Computer Science"

# Stream the response instead of invoking it directly
response = model.stream(query)

# Print the streamed response token by token
for token in response:
    print(token.content, end="", flush=True)
```

    Sure! Here's a light-hearted joke about computer science:
    
    Why did the computer go to the doctor?
    
    Because it had a virus and needed to get "anti-virus"!

# 简介
本程序实现一个简单的可以离线使用的rag系统。

# 技术栈
- sdk：llamaindex
- 向量数据库：chroma
- embeding模型：BAAI/bge-large-zh-v1.5
- LLM模型：qwen/qwen2-1.5b
- 后排序模型：BAAI/bge-reranker-large

# 运行前准备
- 下载代码，安装依赖
```shell
git clone https://github.com/cleanerleon/simple_rag.git
cd simple_rag
poetry install
```
- 下载模型
```shell
huggingface-cli download BAAI/bge-reranker-large
huggingface-cli download BAAI/bge-large-zh-v1.5
huggingface-cli download Qwen/Qwen2-1.5B-Instruct-GGUF qwen2-1_5b-instruct-q5_k_m.gguf --local-dir . --local-dir-use-symlinks False
```

- 放置文档
将文档放在docs目录下

# 运行Demo
```shell
python3 main.py 
load embedding model
load llm model
load vector db
User:Llama2有多少参数
AI: 7B, 13B, 70B, 34B
Explanation: The context mentions that Llama 2 has 7B, 13B, 70B, and 34B variants, which are the parameters released. The answer is a list of these parameters.
User:最少多少
AI: 7B
The context information states that Llama 2 has 7B parameters. Therefore, the minimum number of parameters that Llama 2 has is 7B.
User:Llama2 能商用吗?
AI: 无法确定Llama2是否可用于商业用途，因为没有足够的信息来做出这样的判断。需要更多的上下文和详细信息来评估Llama2的商业可行性。
```

# 问题
- 如果无法运行，将第五行os.environ['HF_HUB_OFFLINE'] = '1'注释后再运行，可能huggingface-cli并没有将全部文件下载成功，待运行成功一次之后再将第5行注释，此后运行将不需要联网。参考[这里](https://github.com/huggingface/transformers/blob/main/docs/source/zh/installation.md)HF_HUB_OFFLINE的解释。
- 如果docs目录添加了新的文档，将chroma_db删除，再启动main.py，会重建向量数据库
- 如果Python集成了SQLite > 3.35，可以注释掉前两行，参考[这里](https://docs.trychroma.com/troubleshooting#sqlite)。

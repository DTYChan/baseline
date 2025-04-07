from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, QueryBundle
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.llms import ChatResponse, LLM, LLMMetadata, ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import _Settings as Settings
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Optional, List, Dict
from peft import PeftModel

# 1. 加载微调过的生成模型和分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device_map="cuda",  
    trust_remote_code=True  # 允许加载自定义代码
)
finetuned_model_path = r"F:\llm\Qwen2.5-1.5B-SFT+RAG\baseline\Qwen_QLoRA"
generation_model = PeftModel.from_pretrained(
    base_model,
    finetuned_model_path  # 训练后的模型路径
)

# 2. 自定义生成模型接口
class CustomFinetunedLLM(LLM):
    def __init__(self, model, tokenizer):
        super().__init__()  # 调用父类的初始化方法
        self._model = model  # 使用下划线前缀表示私有字段
        self._tokenizer = tokenizer

    def complete(self, prompt: str, **kwargs) -> ChatResponse:
        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")  # 将输入数据移动到 GPU
        outputs = self._model.generate(inputs["input_ids"], max_length=2048, temperature=0.7, **kwargs)
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 修改: 返回 ChatResponse 对象时，message 属性应为 ChatMessage 类型
        return ChatResponse(message=ChatMessage(content=generated_text))

    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        prompt = "\n".join([msg.content for msg in messages])  # 使用 msg.content 访问消息内容
        return self.complete(prompt, **kwargs)

    def stream_complete(self, prompt: str, **kwargs) -> ChatResponse:
        return self.complete(prompt, **kwargs)

    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def acomplete(self, prompt: str, **kwargs) -> ChatResponse:
        return self.complete(prompt, **kwargs)

    def achat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def astream_complete(self, prompt: str, **kwargs) -> ChatResponse:
        return self.complete(prompt, **kwargs)

    def astream_chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,  # 设置上下文窗口大小
            num_output=2048,       # 设置最大输出长度
            is_chat_model=True,   # 是否是聊天模型
            is_function_calling_model=False,  # 是否支持函数调用
            model_name="custom_finetuned_model",  # 模型名称
        )
    
# 3. 初始化自定义生成模型
custom_llm = CustomFinetunedLLM(model=generation_model, tokenizer=tokenizer)

# 4. 加载文档并构建索引
documents = SimpleDirectoryReader("data").load_data()  

# 5. 初始化共享的 Embedding 模型
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")  # 使用 sentence-transformers 模型

# 6. 配置 Settings
Settings.llm = custom_llm  # 设置自定义生成模型
Settings.embed_model = embedding_model  # 设置 Embedding 模型

# 7. 构建索引
index = VectorStoreIndex.from_documents(documents)

# 8. 自定义 HyDE 提示模板
hyde_prompt = (
    "Based on the following question, generate a hypothetical document that would answer it:\n"
    "Question: {query_str}\n"
    "Hypothetical Document:"
)

# 9. 自定义 HyDEQueryTransform
class CustomHyDEQueryTransform(HyDEQueryTransform):
    def _run(self, query_bundle: QueryBundle, **kwargs: Any) -> QueryBundle:
        query_str = query_bundle.query_str
        # 使用 complete 方法生成假设文档
        hypothetical_doc = self._llm.complete(self._hyde_prompt.format(query_str=query_str)) 
        # 打印生成的假设文档 
        print(f"HyDE Generated Document: {hypothetical_doc.message}")        
        # 将生成的假设文档的内容包装
        hyde_doc = hypothetical_doc.message.content
        
        # 如果需要包含原始查询，则将原始查询和假设文档组合
        if self._include_original:
            combined_query_str = f"{query_str}\n\nHypothetical Document: {hyde_doc}"
            # 修复：移除不被支持的 embedding_strs 参数
            return QueryBundle(query_str=combined_query_str)
        
        # 否则只返回假设文档
        return QueryBundle(query_str=hyde_doc)

# 10. 使用自定义的 HyDEQueryTransform，并传入自定义提示模板
hyde = CustomHyDEQueryTransform(llm=custom_llm, hyde_prompt=hyde_prompt, include_original=True)

# 11. 创建查询引擎
query_engine = index.as_query_engine()
hyde_query_engine = TransformQueryEngine(query_engine, hyde)

# 12. 用户查询
question = "What can we gain from studying at SJTU?"
response = hyde_query_engine.query(question)
print(f"Retrieved Documents: {response.source_nodes}")  # 打印检索到的文档片段

# 13. 打印结果
print(f"HyDE Query Result: {response}")
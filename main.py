__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP

EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-large"
LLM_MODEL = 'qwen/qwen2-1.5b'
LLM_PATH = 'qwen2-1_5b-instruct-q5_k_m.gguf'
COLLATION_NAME = 'MyLibrary'
DB_PATH = './chroma_db'
DOC_DIR = './docs'

class SimpleRag:
    def __init__(self):
        print('load embedding model')
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL
        )
        print('load llm model')
        llm = LlamaCPP(
            model_path=LLM_PATH, 
            temperature=0,
            verbose=False)
        Settings.llm = llm
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        chroma_collection = chroma_client.get_or_create_collection(COLLATION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # self.index = None
        print('load vector db')
        if chroma_collection.count() == 0:
            documents = SimpleDirectoryReader(
                DOC_DIR, 
                file_extractor={".pdf": PyMuPDFReader()}, 
                recursive=True).load_data()
            splitter = SentenceSplitter(chunk_size=300, chunk_overlap=100)
            nodes = splitter.get_nodes_from_documents(documents)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex(
                nodes, storage_context=storage_context
            )
        else:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store
            )
        fusion_retriever = QueryFusionRetriever(
            [
                self.index.as_retriever(),
                # bm25_retriever
            ],
            similarity_top_k=5,
            num_queries=3,  # 生成 query 数
            use_async=True,
        )

        # 构建单轮 query engine
        reranker = SentenceTransformerRerank(
            model=RERANK_MODEL, top_n=2
        )
        query_engine = RetrieverQueryEngine.from_args(
            fusion_retriever,
            node_postprocessors=[reranker]
        )

        self.chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine, 
        )

    def run(self):
        while True:
            question = input("User:")
            if question.strip() == "":
                break
            response = self.chat_engine.chat(question)
            print(f"AI: {response}")


if __name__ == '__main__':
    rag = SimpleRag()
    rag.run()

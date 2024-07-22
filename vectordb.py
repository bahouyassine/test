import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)
import os

from llama_index.llms.huggingface import HuggingFaceLLM

os.environ['ANTHROPIC_API_KEY'] = ''
llm = Anthropic(temperature=0.0, model='claude-3-haiku-20240307')



# def messages_to_prompt(messages):
#     prompt = ""
#     system_found = False
#     for message in messages:
#         if message.role == "system":
#             prompt += f"<|system|>\n{message.content}<|end|>\n"
#             system_found = True
#         elif message.role == "user":
#             prompt += f"<|user|>\n{message.content}<|end|>\n"
#         elif message.role == "assistant":
#             prompt += f"<|assistant|>\n{message.content}<|end|>\n"
#         else:
#             prompt += f"<|user|>\n{message.content}<|end|>\n"

#     # trailing prompt
#     prompt += "<|assistant|>\n"

#     if not system_found:
#         prompt = (
#             "<|system|>\nYou are a helpful AI assistant.<|end|>\n" + prompt
#         )

#     return prompt


# llm = HuggingFaceLLM(
#     model_name="microsoft/Phi-3-mini-128k-instruct",
#     model_kwargs={
#         "trust_remote_code": True,
#     },
#     generate_kwargs={"do_sample": True, "temperature": 0.1},
#     tokenizer_name="microsoft/Phi-3-mini-128k-instruct",
#     query_wrapper_prompt=(
#         "<|system|>\n"
#         "You are a helpful AI assistant.<|end|>\n"
#         "<|user|>\n"
#         "{query_str}<|end|>\n"
#         "<|assistant|>\n"
#     ),
#     messages_to_prompt=messages_to_prompt,
#     is_chat_model=True,
# )

embed_model = HuggingFaceEmbedding(model_name="Alibaba-NLP/gte-large-en-v1.5",trust_remote_code=True)



Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512



def create_index(documents_folder_path, vector_db_path):
    documents = SimpleDirectoryReader(documents_folder_path).load_data()

    # initialize client, setting path to save data
    db = chromadb.PersistentClient(path=vector_db_path)

    # create collection
    chroma_collection = db.get_or_create_collection("quickstart")

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # create your index
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

def load_index(vector_db_path):
    
    # initialize client
    db = chromadb.PersistentClient(path=vector_db_path)

    # get collection
    chroma_collection = db.get_or_create_collection("quickstart")

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # load your index from stored vectors
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    return index


def answer_query(query,vector_db_path):
    index = load_index(vector_db_path)
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(query)
    return response



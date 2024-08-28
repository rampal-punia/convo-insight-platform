from decouple import config
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFaceEndpoint
# from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
HUGGINGFACE_API_TOKEN = config('HUGGINGFACEHUB_API_TOKEN')


def configure_llm(model_name, **kwargs):
    '''Create LLM with configurations'''
    huggingface_models = {
        "Mistral-7B": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
        "Mixtral-8x7B": "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Mistral-Nemo": "https://api-inference.huggingface.co/models/mistralai/Mistral-Nemo-Instruct-2407",
    }

    if model_name in huggingface_models:
        return HuggingFaceEndpoint(
            endpoint_url=huggingface_models[model_name],
            task='text-generation',
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def create_chain(prompt, llm, run_name):
    '''Creating LLM chain'''
    output_parser = StrOutputParser()
    return prompt | llm.with_config({'run_name': 'model'}) | output_parser.with_config({'run_name': run_name})


chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ('human', "Conversation history:\n{history}\n\nNew User message: {input}"),
    ("human", "Now, respond to the new message.")
])

doc_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant helping to answer questions based on a given document. Use the following context to answer the user's question. If you cannot answer the question based on the context, say that you don't have enough information to answer accurately."),
    ('human', "Conversation history:\n{history}\n\nNew User message: {input}"),
    ('human', "Related Context:\n{context}"),
    ("human", "Now, respond to the new message.")
])

# Example usage
# callbacks = [StreamingStdOutCallbackHandler()]
llm = configure_llm(
    "Mixtral-8x7B",
    max_new_tokens=512,
    top_k=20,
    top_p=0.85,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    # callbacks=callbacks,
    streaming=True,
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
)

chain = create_chain(chat_prompt, llm, "Assistant")

doc_chain = create_chain(doc_prompt, llm, "Assistant")

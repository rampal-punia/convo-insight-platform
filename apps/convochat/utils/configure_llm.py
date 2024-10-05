from django.conf import settings
import aiohttp
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser


class LLMConfig:
    HUGGINGFACE_MODELS = {
        "Mistral-7B": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
        "Mistral-Code-7B": "https://api-inference.huggingface.co/models/mistralai/Mamba-Codestral-7B-v0.1",
        "Mistral-Nvidia-7B": "https://api-inference.huggingface.co/models/nvidia/Mistral-NeMo-Minitron-8B-Base",
        "Mixtral-8x7B-I": "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Mixtral-8x7B": "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-v0.1",
        "Mixtral-8x22B": "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x22B-Instruct-v0.1",
        "Mistral-Nemo": "https://api-inference.huggingface.co/models/mistralai/Mistral-Nemo-Instruct-2407",
    }
    HUGGINGFACE_MODELS_REPO = {
        "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
        "Mistral-Code-7B": "mistralai/Mamba-Codestral-7B-v0.1",
        "Mistral-Nvidia-7B": "nvidia/Mistral-NeMo-Minitron-8B-Base",
        "Mixtral-8x7B-I": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Mixtral-8x7B": "mistralai/Mixtral-8x7B-v0.1",
        "Mixtral-8x22B": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "Mistral-Nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    }

    @classmethod
    def configure_llm(cls, model_name, **kwargs):
        if model_name not in cls.HUGGINGFACE_MODELS_REPO:
            raise ValueError(f"Unsupported model: {model_name}")

        return HuggingFaceEndpoint(
            # endpoint_url=cls.HUGGINGFACE_MODELS[model_name],
            repo_id=cls.HUGGINGFACE_MODELS_REPO[model_name],
            task='text-generation',
            huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN,
            **kwargs
        )

    @staticmethod
    def get_llm(model_name="Mixtral-8x7B-I", temperature=0.03, tokens=512, top_k=25, top_p=0.85,
                typical_p=0.95, repetition_penalty=1.03, is_streaming=True):
        return LLMConfig.configure_llm(
            model_name=model_name,
            max_new_tokens=tokens,
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            streaming=is_streaming,
        )


class CustomPromptTemplates:
    @staticmethod
    def get_chat_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ('human',
             "Conversation history:\n{history}\n\nNew User message: {input}"),
            ("human", "Now, respond to the new message.")
        ])

    @staticmethod
    def get_summarizer_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful summarizer."),
            ("human", "Now, summarize these given paragraphs: {input}.")
        ])

    @staticmethod
    def get_doc_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant helping to answer questions based on a given document. Use the following context to answer the user's question. If you cannot answer the question based on the context, say that you don't have enough information to answer accurately."),
            ('human',
             "Related Context:\n{context}\n\nNew User message: {input}"),
            ("human", "Now, respond to the new message.")
        ])

    @staticmethod
    def get_orders_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful customer support assistant for an e-commerce company."),
            ('human', """
            Conversation history:
            {history}
            
            Order information:
            {order_dict}
            
            New User message: {input}
            """),
            ("human", "Now, respond to the new message, taking into account the order details if relevant. Use the full status descriptions when referring to order status.")
        ])

    @staticmethod
    def get_doc_prompt_with_history():
        return ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant helping to answer, in short (150 to 200 words only), the questions based on a given document. Use the following context to answer the user's question. If you cannot answer the question based on the context, say that you don't have enough information to answer accurately."),
            ('human',
             "Conversation history:\n{history}\n\nNew User message: {input}"),
            ('human', "Related Context:\n{context}"),
            ("human", "Now, respond to the new message.")
        ])


class ChainBuilder:
    @staticmethod
    def create_chat_chain(prompt, llm, run_name):
        output_parser = StrOutputParser()
        return prompt | llm.with_config({'run_name': 'model'}) | output_parser.with_config({'run_name': run_name})

    @staticmethod
    def create_order_chat_chain(prompt, llm, run_name):
        output_parser = StrOutputParser()
        return prompt | llm.with_config({'run_name': 'model'}) | output_parser.with_config({'run_name': run_name})

    @staticmethod
    def create_qa_chain(retriever, prompt, llm, output_parser):
        return (
            {
                "context": retriever,
                # "context": retriever | DocumentUtils.format_docs,
                "input": RunnablePassthrough(),
            }
            | prompt
            | llm.with_config({'run_name': 'model'})
            | output_parser
        )

    @staticmethod
    def create_doc_chain(retrieved_docs, prompt, llm, run_name):
        output_parser = StrOutputParser()
        chain = (
            RunnablePassthrough.assign(context=retrieved_docs)
            | prompt
            | llm.with_config({'run_name': 'model'})
            | output_parser.with_config({'run_name': run_name})
        )
        return RunnableWithMessageHistory(
            chain,
            RedisChatMessageHistory,
            input_messages_key="question",
            history_messages_key="chat_history",
        )


class LLMInvoker:
    @staticmethod
    def invoke_llm(memory_chain, user_question: str = 'What is modern science', session_id='123456789'):
        return memory_chain.invoke(user_question)
        # return memory_chain.invoke(
        #     {"question": user_question},
        #     config={"configurable": {"session_id": session_id}},
        # )


class DocumentUtils:
    @staticmethod
    def get_sources(docs):
        return [", ".join([doc.metadata["source"] for doc in docs])]

    @staticmethod
    def format_docs(docs):
        # DocumentUtils.get_sources(docs)
        return "\n\n".join([doc.page_content for doc in docs])


def main(context=None):
    prompt = CustomPromptTemplates.get_chat_prompt()
    llm = LLMConfig.get_llm()
    if context:
        return ChainBuilder.create_doc_chain(
            retrieved_docs=context,
            prompt=prompt,
            llm=llm,
            run_name='Assistant'
        )
    return ChainBuilder.create_chat_chain(
        prompt=prompt,
        llm=llm,
        run_name='Assistant'
    )


def order_main():
    prompt = CustomPromptTemplates.get_orders_prompt()
    llm = LLMConfig.get_llm()
    return ChainBuilder.create_order_chat_chain(
        prompt=prompt,
        llm=llm,
        run_name='Assistant'
    )


chain = ChainBuilder.create_chat_chain(
    prompt=CustomPromptTemplates.get_chat_prompt(),
    llm=LLMConfig.get_llm(),
    run_name='Assistant'
)


async def generate_title(conversation_content):
    API_URL_TITLE = "https://api-inference.huggingface.co/models/czearing/article-title-generator"
    headers = {"Authorization": f"Bearer {settings.HUGGINGFACEHUB_API_TOKEN}"}
    async with aiohttp.ClientSession() as session:
        async with session.post(
                API_URL_TITLE,
                headers=headers,
                json={
                    "inputs": conversation_content,
                    "parameters": {"max_length": 50, "min_length": 10}
                }) as response:
            result = await response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0]['generated_text']
            else:
                return "Untitled Conversation"


async def generate_summary(conversation_content):
    API_URL_SUMMARY = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {settings.HUGGINGFACEHUB_API_TOKEN}"}
    async with aiohttp.ClientSession() as session:
        async with session.post(
                API_URL_SUMMARY,
                headers=headers,
                json={
                    "inputs": conversation_content,
                    "parameters": {"max_length": 50, "min_length": 10}
                }) as response:
            result = await response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0]['generated_summary']
            else:
                return "Untitled Conversation"

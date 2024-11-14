from django.conf import settings
import aiohttp
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from typing import Literal, Optional, Union


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

    OPENAI_MODELS = {
        'gpt-4o-mini': 'gpt-4o-mini',
        "gpt-4": "gpt-4",
        "gpt-4-turbo": "gpt-4-0125-preview",
        "gpt-4-32k": "gpt-4-32k",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k",
    }

    @classmethod
    def configure_huggingface_llm(cls, model_name: str, **kwargs) -> HuggingFaceEndpoint:
        """Configure and return a HuggingFace model endpoint."""
        if model_name not in cls.HUGGINGFACE_MODELS_REPO:
            raise ValueError(f"Unsupported HuggingFace model: {model_name}")

        return HuggingFaceEndpoint(
            repo_id=cls.HUGGINGFACE_MODELS_REPO[model_name],
            task='text-generation',
            huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN,
            **kwargs
        )

    @classmethod
    def configure_openai_llm(cls, model_name: str, **kwargs) -> ChatOpenAI:
        """Configure and return an OpenAI chat model."""
        if model_name not in cls.OPENAI_MODELS:
            raise ValueError(f"Unsupported OpenAI model: {model_name}")

        return ChatOpenAI(
            model=cls.OPENAI_MODELS[model_name],
            openai_api_key=settings.OPENAI_API_KEY,
            **kwargs
        )

    @staticmethod
    def get_llm(
        model_name: str = "Mixtral-8x7B-I",
        model_provider: Optional[Literal["openai", "huggingface"]] = None,
        temperature: float = 0.03,
        tokens: int = 512,
        top_k: int = 25,
        top_p: float = 0.85,
        typical_p: float = 0.95,
        repetition_penalty: float = 1.03,
        is_streaming: bool = True,
        **kwargs
    ) -> Union[HuggingFaceEndpoint, ChatOpenAI]:
        """
        Get a configured LLM based on the model name and provider.

        Args:
            model_name: Name of the model to use
            model_provider: Explicitly specify the provider ("openai" or "huggingface")
            temperature: Controls randomness in the output
            tokens: Maximum number of tokens to generate
            top_k: Number of highest probability vocabulary tokens to keep
            top_p: Cumulative probability cutoff for token selection
            typical_p: Mass of probability distribution to consider
            repetition_penalty: Penalty for token repetition
            is_streaming: Whether to enable streaming response
            **kwargs: Additional arguments to pass to the model

        Returns:
            Configured LLM instance
        """
        # Determine provider if not explicitly specified
        if not model_provider:
            if model_name in LLMConfig.OPENAI_MODELS:
                model_provider = "openai"
            elif model_name in LLMConfig.HUGGINGFACE_MODELS_REPO:
                model_provider = "huggingface"
            else:
                raise ValueError(
                    f"Unknown model: {model_name}. Please specify model_provider.")

        if model_provider == "openai":
            return LLMConfig.configure_openai_llm(
                model_name=model_name,
                temperature=temperature,
                streaming=is_streaming,
                max_tokens=tokens,
                **kwargs
            )
        else:  # huggingface
            return LLMConfig.configure_huggingface_llm(
                model_name=model_name,
                max_new_tokens=tokens,
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                streaming=is_streaming,
                **kwargs
            )

    @staticmethod
    def list_available_models(provider: Optional[Literal["openai", "huggingface"]] = None) -> dict:
        """List available models, optionally filtered by provider."""
        if provider == "openai":
            return LLMConfig.OPENAI_MODELS
        elif provider == "huggingface":
            return LLMConfig.HUGGINGFACE_MODELS_REPO
        else:
            return {
                "openai": LLMConfig.OPENAI_MODELS,
                "huggingface": LLMConfig.HUGGINGFACE_MODELS_REPO
            }


class CustomPromptTemplates:
    @staticmethod
    def get_chat_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "Direct and efficient assistant. Provide complete, accurate responses while maintaining brevity. Focus on addressing the specific query."),
            ("human", "Context:\n{history}\n\nUser: {input}"),
            ("human", "Respond to the query while maintaining conversation context.")
        ])

    @staticmethod
    def get_summarizer_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful summarizer."),
            ("human", "Now, summarize these conversations: {input}.")
        ])

    @staticmethod
    def get_doc_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "Extract relevant information from provided context to answer questions accurately. Acknowledge knowledge gaps when context is insufficient."),
            ("human", "Context:\n{context}\n\nQuestion: {input}"),
            ("human", "Provide direct, evidence-based response.")
        ])

    @staticmethod
    def get_orders_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "E-commerce support agent. Provide accurate, empathetic responses. Reference order details when relevant. Request clarification if needed."),
            ("human", """Context:
                {history}

                Order Data:
                {order_dict}

                Query: {input}"""),
            ("human", "Respond professionally with relevant order details or general assistance as appropriate.")
        ])

    @staticmethod
    def get_doc_prompt_with_history():
        return ChatPromptTemplate.from_messages([
            ("system", "Provide concise (150-200 words) answers using provided context. Reference relevant information and indicate knowledge gaps. Maintain conversation continuity."),
            ("human", """History:
                {history}

                Query: {input}

                Context:
                {context}"""),
            ("human", "Respond concisely with context-based information.")
        ])


class ChainBuilder:
    @staticmethod
    def create_chat_chain(prompt, llm, run_name):
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


def get_general_assistant_llm(model_name='gpt-4o-mini', model_provider='openai'):
    """Get LLM for the order support agent

    Args:
        model_name (str, optional): LLM Name. Defaults to 'gpt-4o-mini'. For huggingface Select: model_name="Mixtral-8x7B-I"
        model_provider (str, optional): LLM Service Provider. Defaults to 'openai'. for huggingface select: 'huggingface'

    Returns:
        LLM: LLM either from openai or huggingface
    """
    prompt = CustomPromptTemplates.get_orders_prompt()
    llm = LLMConfig.get_llm(
        model_name=model_name,
        model_provider=model_provider,
        temperature=0.1,
    )
    return ChainBuilder.create_chat_chain(
        prompt=prompt,
        llm=llm,
        run_name='Assistant'
    )


def get_chat_llm(model_name='gpt-4o-mini', model_provider='openai'):
    """Get LLM for the order support agent

    Args:
        model_name (str, optional): LLM Name. Defaults to 'gpt-4o-mini'. For huggingface Select: model_name="Mixtral-8x7B-I"
        model_provider (str, optional): LLM Service Provider. Defaults to 'openai'. for huggingface select: 'huggingface'

    Returns:
        LLM: LLM either from openai or huggingface
    """
    prompt = CustomPromptTemplates.get_chat_prompt()
    llm = LLMConfig.get_llm(
        model_name=model_name,
        model_provider=model_provider,
        temperature=0.1,
    )
    return ChainBuilder.create_chat_chain(
        prompt=prompt,
        llm=llm,
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


# List available models
# all_models = LLMConfig.list_available_models()
# openai_models = LLMConfig.list_available_models(provider="openai")

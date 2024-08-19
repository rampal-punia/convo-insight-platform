# apps/convochat/utils/configure_llm.py

from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from llms.fine_tuning.llm_fine_tuner import LLMFineTuner
from django.conf import settings

# def configure_llm(repo_id, max_new_tokens, top_k, temperature, repetition_penalty, use_fine_tuned=True):
#     '''Create LLM with configurations'''
#     if use_fine_tuned:
#         fine_tuner = LLMFineTuner(
#             model_name=repo_id,
#             dataset_path="path/to/your/annotated/dataset",
#             output_dir="fine_tuned_output"
#         )
#         fine_tuner.load_fine_tuned_model()
#         model_path = "fine_tuned_output/fine_tuned_model"
#     else:
#         model_path = repo_id

#     return HuggingFaceEndpoint(
#         repo_id=model_path,
#         task='text-generation',
#         max_new_tokens=max_new_tokens,
#         top_k=top_k,
#         temperature=temperature,
#         repetition_penalty=repetition_penalty
#     )


def configure_llm(model_name, **kwargs):
    '''Create LLM with configurations'''
    huggingface_models = {
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
        "llama3-8b": "meta-llama/Meta-Llama-3-8B",
        "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
        "llama2-70b": "meta-llama/Llama-2-70b-chat-hf",
        "finbert-tone": "nickmuchi/finbert-tone-finetuned-finance-topic-classification",
    }

    if model_name in huggingface_models:
        return HuggingFaceEndpoint(
            repo_id=huggingface_models[model_name],
            task='text-generation',
            **kwargs
        )
    if model_name == "fine_tuned":
        return HuggingFaceEndpoint(
            repo_id=settings.FINE_TUNED_MODEL_DIR,
            task='text-generation',
            **kwargs
        )
    # elif model_name == "gpt-3.5-turbo":
    #     return ChatOpenAI(model_name="gpt-3.5-turbo", **kwargs)
    # elif model_name == "gpt-4":
    #     return ChatOpenAI(model_name="gpt-4", **kwargs)
    # elif model_name == "claude-2":
    #     return ChatAnthropic(model_name="claude-2", **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def create_chain(prompt, llm, run_name):
    '''Creating LLM chain'''
    output_parser = StrOutputParser()
    return prompt | llm.with_config({'run_name': 'model'}) | output_parser.with_config({'run_name': run_name})


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ('human', "Conversation history:\n{history}\n\nNew User message: {input}"),
    ("human", "Now, respond to the new message.")
])

# Example usage
llm = configure_llm(
    "mistral-8x7b",
    temperature=0.5,
    top_k=25,
    max_new_tokens=1000
)

chain = create_chain(prompt, llm, "Assistant")


# llm = configure_llm(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#     max_new_tokens=400,
#     top_k=25,
#     temperature=0.2,
#     repetition_penalty=1.5
# )

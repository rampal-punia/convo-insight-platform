from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser


def configure_llm(repo_id, max_new_tokens, top_k, temperature, repetition_penalty):
    '''Create LLM with configurations'''
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        task='text-generation',
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty
    )


def create_chain(prompt, llm, run_name):
    '''Creating LLM chain'''
    output_parser = StrOutputParser()
    return prompt | llm.with_config({'run_name': 'model'}) | output_parser.with_config({'run_name': run_name})


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ('human', "Conversation history:\n{history}\n\nNew User message: {input}"),
    ("human", "Now, respond to the new message.")
])

llm = configure_llm(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    max_new_tokens=400,
    top_k=25,
    temperature=0.2,
    repetition_penalty=1.5
)

chain = create_chain(prompt, llm, "Assistant")

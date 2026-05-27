from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
from asgiref.sync import sync_to_async
from .text_classification_vector_store import PGVectorStoreTextClassification


class RAGProcessorTextClassification:
    def __init__(self, vector_store: PGVectorStoreTextClassification, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.setup_prompts()

    def setup_prompts(self):
        """Setup task-specific prompts for RAG"""
        self.prompts = {
            'SE': ChatPromptTemplate.from_messages([
                ("system", "You are a sentiment analysis expert. Use the following similar examples to guide your analysis:"),
                ("human",
                 "Similar examples:\n{context}\n\nAnalyze the sentiment of: {query}"),
            ]),
            'IN': ChatPromptTemplate.from_messages([
                ("system", "You are an intent recognition specialist. If you are given intent and the text you can identify the intent out of it and provide explanation for that selection of intent. Use the following similar examples to guide your analysis:"),
                ("human",
                 "Similar examples:\n{context}\n\nIdentify the intent of this text: {query} and provide explanation."),
            ]),
            'TO': ChatPromptTemplate.from_messages([
                ("system", "You are a topic classification expert. Use the following similar examples to guide your classification:"),
                ("human",
                 "Similar examples:\n{context}\n\nClassify the topic of: {query}"),
            ])
        }

    @sync_to_async
    def get_similar_docs(self, text: str, task: str):
        '''Wrapper for synchronous similarity search'''
        document = self.vector_store.similarity_search(text, task)
        return [(doc.metadata, doc.content) for doc in document]
        # return [(doc.metadata['intent_name'], doc.content) for doc in document]

    async def process(self, text: str, task: str) -> Dict:
        '''Process text using RAG'''

        # Retrive similar examples
        docs = await self.get_similar_docs(text, task)
        context = "\n\n".join([f"{k} : {v}" for k, v in docs])

        # Get prompt for task
        prompt = self.prompts[task]
        print(
            f"prompt for rag {task} is: {prompt.format(context=context, query=text)}")

        # Generate response
        response = await self.llm.ainvoke(
            prompt.format(context=context, query=text)
        )
        print("LLM reponse is: ", response)

        # Parse response based on task
        result = self._parse_response(response.content, task)
        return result

    def _parse_response(self, content: str, task: str) -> Dict:
        "Parse LLM response based on task"
        result = {
            'label': None,
            'score': 0.8,  # Default confidence
            'explanation': content
        }

        # Extract task-specific information
        if task == 'SE':
            # Parse sentiment response
            if 'POSITIVE' in content.upper():
                result['label'] = 'POSITIVE'
            elif 'NEGATIVE' in content.upper():
                result['label'] = 'NEGATIVE'
            else:
                result['label'] = 'NEUTRAL'

        elif task == 'IN':
            # Parse intent response
            result['label'] = content.split(
                "'")[3] if "'" in content else content

        elif task == 'TO':
            # Parse topic response
            result['label'] = content.split(
                "'")[3] if "'" in content else content

        return result

from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
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
                ("system", "You are an intent recognition specialist. Use the following similar examples to guide your analysis:"),
                ("human",
                 "Similar examples:\n{context}\n\nIdentify the intent of: {query}"),
            ]),
            'TO': ChatPromptTemplate.from_messages([
                ("system", "You are a topic classification expert. Use the following similar examples to guide your classification:"),
                ("human",
                 "Similar examples:\n{context}\n\nClassify the topic of: {query}"),
            ])
        }

    async def process(self, text: str, task: str) -> Dict:
        "Process text using RAG"

        # Retrive similar examples
        similar_docs = self.vector_store.similarity_search(text, task)
        context = "\n\n".join([doc.content for doc in similar_docs])

        # Get prompt for task
        prompt = self.prompts[task]

        # Generate response
        response = await self.llm.ainvoke(
            prompt.format(context=context, query=text)
        )

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
                "'")[1] if "'" in content else content

        elif task == 'TO':
            # Parse topic response
            result['label'] = content.split(
                "'")[1] if "'" in content else content

        return result

from typing import Any, Callable, List, Optional, cast, Dict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, Tool
from typing_extensions import Annotated

from configuration import Configuration


async def tavily_search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


async def duckduckgo_search(
        query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> List[Dict[str, str]]:
    """Search for general web results using DuckDuckGo
    This function performs a search using the DuckDuckGo search engine, which is known
    for its privacy-focused approach and unbiased results.

    Args:
        query: The search query string
        config: Configuration including max_search_results

    Returns:
        List of search results, each containing title, link, and snippet
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize DuckDuckGo search wrapper with configuration
    search = DuckDuckGoSearchResults(
        output_format="list",
        num_results=configuration.max_search_results,
    )
    # Perform the search
    result = await search.ainvoke(query)
    return cast(list[dict[str, Any]], result)


TOOLS: List[Callable[..., Any]] = [duckduckgo_search]


if __name__ == '__main__':
    import asyncio

    async def test_searches():
        query = "what is artificial intelligence?"
        print("\nTesting Tavily Search:")
        tavily_results = await tavily_search(query, config={})
        print(f"Found {len(tavily_results)} results")

        print("\nTesting DuckDuckGo Search:")
        ddg_results = await duckduckgo_search(query, config={})
        print(f"Found {len(ddg_results)} results")

    asyncio.run(test_searches())

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from faq_service.application.models.agent_models import SearchResult
from faq_service.domain.validation import validate_vector


class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


def create_vector_search_tool(settings, db, embeddings):
    @tool(args_schema=SearchInput, response_format="content_and_artifact")
    async def vector_search(query: str):
        """Find FAQ passages relevant to a question using cosine vector search."""
        vector = validate_vector(
            await embeddings.aembed_query(query),
            settings.embedding_dimension,
        )
        chunks = await db.documents.search(vector, settings.search_top_k)
        result = SearchResult(query=query, chunks=chunks)
        return result.model_dump_json(), result

    return vector_search

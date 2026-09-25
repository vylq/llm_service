from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from faq_service.settings.app_settings import Settings


def chat_model(settings: Settings):
    if settings.llm_provider == "ollama":
        return ChatOllama(
            model=settings.ollama_llm_model,
            base_url=settings.chat_base_url,
            temperature=settings.llm_temperature,
            client_kwargs={"timeout": settings.llm_timeout_seconds},
        )
    key = settings.llm_api_key or settings.openrouter_api_key
    if not key.get_secret_value():
        raise ValueError("Set OPENROUTER_API_KEY or LLM_API_KEY in .env")
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.chat_base_url,
        api_key=key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        use_responses_api=False,
        extra_body={"provider": {"require_parameters": True}}
        if settings.llm_require_parameters
        else None,
    )


def embedding_model(settings: Settings):
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.embeddings_base_url,
            client_kwargs={"timeout": settings.llm_timeout_seconds},
        )
    key = settings.embedding_api_key or settings.openrouter_api_key
    if not key.get_secret_value():
        raise ValueError("Set OPENROUTER_API_KEY or EMBEDDING_API_KEY in .env")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.embeddings_base_url,
        api_key=key,
        check_embedding_ctx_length=False,  # Send text, not provider-specific token IDs.
        model_kwargs={"encoding_format": "float"},
        chunk_size=settings.embedding_batch_size,
        request_timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )

import math

from faq_service.domain.errors import ServiceError


def validate_vector(vector: list[float], dimension: int) -> list[float]:
    if len(vector) != dimension or not all(math.isfinite(x) for x in vector):
        raise ServiceError("invalid_embedding", "Некорректная размерность или значения embedding.")
    if not any(vector):
        raise ServiceError("invalid_embedding", "Получен нулевой embedding.")
    return vector

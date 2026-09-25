from pydantic import BaseModel

# Domain data has no dependencies on API or orchestration.


class Source(BaseModel):
    file: str
    line: int
    file_sha256: str


class Document(BaseModel):
    id: str
    title: str
    body: str
    sources: list[Source]


class RetrievedChunk(BaseModel):
    id: str
    document_id: str
    title: str
    text: str
    distance: float


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    quote: str

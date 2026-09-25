"""Local OpenRouter-compatible stub for transport/Compose tests, not a real LLM."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def response_for(path, body):
    if path.endswith("/embeddings"):
        texts = body["input"]
        if isinstance(texts, str):
            texts = [texts]
        assert all(isinstance(t, str) for t in texts), "Embedding input must be raw text"
        return {
            "object": "list",
            "model": body["model"],
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": [1.0, 0.0, 0.0] if "выплат" in text.lower() else [0.0, 1.0, 0.0],
                }
                for i, text in enumerate(texts)
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
    assert path.endswith("/chat/completions"), path
    schema = body.get("response_format", {}).get("json_schema", {}).get("name")
    message = {"role": "assistant", "content": "Поиск завершён."}
    finish = "stop"
    if schema == "ModerationResult":
        message["content"] = json.dumps({"decision": "allow", "reason": "FAQ"})
    elif schema == "WriterResult":
        payload = json.loads(body["messages"][1]["content"])
        chunks = payload["evidence"]
        result = {"status": "no_answer", "answer": "Нет сведений.", "citations": []}
        if chunks:
            chunk = chunks[0]
            result = {
                "status": "answered",
                "answer": chunk["text"] + " [1]",
                "citations": [
                    {
                        "document_id": chunk["document_id"],
                        "chunk_id": chunk["id"],
                        "quote": chunk["text"][:30],
                    }
                ],
            }
        message["content"] = json.dumps(result, ensure_ascii=False)
    elif not any(m["role"] == "tool" for m in body["messages"]):
        assert body.get("tools"), "RAG must provide tool schema"
        finish = "tool_calls"
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {
                        "name": "vector_search",
                        "arguments": json.dumps({"query": "Как повторить выплату?"}),
                    },
                }
            ],
        }
    return {
        "id": "test-completion",
        "object": "chat.completion",
        "created": 1,
        "model": body["model"],
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        result = json.dumps(response_for(self.path, body), ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(result)))
        self.end_headers()
        self.wfile.write(result)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 9000), Handler).serve_forever()

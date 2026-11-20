"""Provider adapters keep wire formats out of the experiment runner."""

import base64
import binascii
import os
from urllib.parse import parse_qs, quote, urlsplit

import httpx

from .endpoints import Endpoint, Route, configured_routes


def bedrock_key_region(key: str) -> str | None:
    prefix = "bedrock-api-key-"
    if not key.startswith(prefix):
        return None
    try:
        token = base64.b64decode(key[len(prefix):], validate=True).decode()
        credential = parse_qs(urlsplit("https://" + token).query).get("X-Amz-Credential", [""])[0]
        fields = credential.split("/")
        return fields[2] if len(fields) == 5 and fields[3] == "bedrock" else None
    except (ValueError, UnicodeError, binascii.Error):
        return None


class GroqClient:
    def routes(self, endpoint: Endpoint) -> list[Route]:
        return configured_routes()

    async def models(self, http: httpx.AsyncClient, endpoint: Endpoint, route: Route) -> set[str]:
        response = await http.get(f"{endpoint.base_url}/models", headers=route.headers())
        if not response.is_success:
            raise ValueError(f"{route.label} model catalog returned HTTP {response.status_code}")
        return {item["id"] for item in response.json()["data"]}

    def payload(self, endpoint: Endpoint, prompt: str, schema: dict) -> dict:
        request = {"model": endpoint.model_id, "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_completion_tokens": endpoint.max_tokens}
        if endpoint.reasoning:
            request["reasoning_effort"] = endpoint.reasoning
            request["include_reasoning"] = False
        if endpoint.structured_output:
            request["response_format"] = schema
        return request

    async def complete(self, http, endpoint, route, payload) -> httpx.Response:
        return await http.post(f"{endpoint.base_url}/chat/completions", headers=route.headers(), json=payload)

    def normalize(self, body):
        return body


class BedrockClient:
    def routes(self, endpoint: Endpoint) -> list[Route]:
        key = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "").strip()
        if not key:
            raise ValueError("Set AWS_BEARER_TOKEN_BEDROCK in .env for Bedrock on-demand calls")
        key_region = bedrock_key_region(key)
        if key_region and key_region != endpoint.region:
            raise ValueError(f"Bedrock key belongs to {key_region}; requested region is {endpoint.region}")
        label = f"bedrock-{endpoint.region}"
        return [Route(label, label, key)]

    async def models(self, http: httpx.AsyncClient, endpoint: Endpoint, route: Route) -> set[str]:
        # A runtime bearer may not have catalog permissions. Converse validates model access.
        return {endpoint.model_id}

    def payload(self, endpoint: Endpoint, prompt: str, schema: dict) -> dict:
        request = {"messages": [{"role": "user", "content": [{"text": prompt}]}],
                   "inferenceConfig": {"maxTokens": endpoint.max_tokens, "temperature": 0}}
        if endpoint.reasoning:
            request["additionalModelRequestFields"] = {"reasoning_effort": endpoint.reasoning}
        return request

    async def complete(self, http, endpoint, route, payload) -> httpx.Response:
        model = quote(endpoint.model_id, safe="")
        return await http.post(f"{endpoint.base_url}/model/{model}/converse",
                               headers=route.headers(), json=payload)

    def normalize(self, body):
        if not isinstance(body, dict):
            return body
        output = body.get("output")
        if not isinstance(output, dict):
            return body
        message = output.get("message") or {}
        blocks = message.get("content") or []
        visible = "\n".join(block["text"] for block in blocks if isinstance(block.get("text"), str))
        # Some reasoning providers include a thinking block inside a visible text field.
        if visible.lstrip().startswith("<think>") and "</think>" in visible:
            visible = visible.split("</think>", 1)[1].strip()
        usage = body.get("usage") or {}
        return {"choices": [{"message": {"content": visible}, "finish_reason": body.get("stopReason")}],
                "usage": {"prompt_tokens": usage.get("inputTokens"),
                          "completion_tokens": usage.get("outputTokens"),
                          "total_tokens": usage.get("totalTokens")},
                "bedrock": body}


GROQ = GroqClient()
BEDROCK = BedrockClient()


def client_for(endpoint: Endpoint) -> GroqClient | BedrockClient:
    return BEDROCK if endpoint.region else GROQ

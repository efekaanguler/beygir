"""API models, credential routing and server-reported limits."""

import os
import re
from dataclasses import dataclass, field

import httpx

GROQ_URL = "https://api.groq.com/openai/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = ("gpt20", "gpt120", "qwen")


@dataclass(frozen=True)
class Limits:
    rpm: int = 30
    rpd: int = 1000
    tpm: int = 8000
    tpd: int = 200_000


@dataclass(frozen=True)
class Endpoint:
    alias: str
    model_id: str
    base_url: str = GROQ_URL
    provider: str = ""
    reasoning: str = "low"
    limits: Limits = Limits()
    max_tokens: int = 4096
    structured_output: bool = True
    token_estimate: int = 2600

    def signature(self) -> dict:
        return {"alias": self.alias, "model_id": self.model_id, "api": self.base_url,
                "provider": self.provider, "reasoning": self.reasoning,
                "temperature": 0, "max_completion_tokens": self.max_tokens,
                "structured_output": self.structured_output}

    def payload(self, prompt: str, schema: dict) -> dict:
        request = {"model": self.model_id, "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]}
        if self.base_url == GROQ_URL:
            request["max_completion_tokens"] = self.max_tokens
            if self.reasoning:
                request["reasoning_effort"] = self.reasoning
                request["include_reasoning"] = False
        else:
            request["max_tokens"] = self.max_tokens
            request["provider"] = {"only": [self.provider], "allow_fallbacks": False}
        if self.structured_output:
            request["response_format"] = schema
        return request


@dataclass(frozen=True)
class Route:
    label: str
    quota_group: str
    key: str = field(repr=False)

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"}


def catalog() -> dict[str, Endpoint]:
    provider = os.getenv("OPENROUTER_PROVIDER", "").strip()
    return {
        "gpt20": Endpoint("gpt20", "openai/gpt-oss-20b"),
        "gpt120": Endpoint("gpt120", "openai/gpt-oss-120b"),
        "qwen": Endpoint("qwen", "qwen/qwen3.8-27b", reasoning="none", max_tokens=512, token_estimate=750),
        "allam": Endpoint("allam", "allam-2-7b", reasoning="",
                          limits=Limits(rpd=7000, tpm=6000, tpd=500_000), max_tokens=512, structured_output=False, token_estimate=900),
        "llama": Endpoint("llama", "meta-llama/llama-3.3-70b-instruct", OPENROUTER_URL,
                          provider=provider, reasoning="",
                          limits=Limits(rpd=10000, tpm=8000, tpd=4_000_000), max_tokens=512, structured_output=False, token_estimate=900),
    }


def configured_routes(endpoint: Endpoint) -> list[Route]:
    if endpoint.base_url == OPENROUTER_URL:
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key or not endpoint.provider:
            raise ValueError("Llama needs OPENROUTER_API_KEY and OPENROUTER_PROVIDER in .env")
        return [Route("openrouter", f"openrouter:{endpoint.provider}", key)]
    numbers = sorted(int(match[1]) for name in os.environ
                     if (match := re.fullmatch(r"GROQ_API_KEY_(\d+)", name)))
    routes = []
    keys = set()
    for number in numbers:
        key = os.getenv(f"GROQ_API_KEY_{number}", "").strip()
        group = os.getenv(f"GROQ_QUOTA_GROUP_{number}", "").strip() or "groq"
        if key and key not in keys:
            routes.append(Route(f"groq-{number}", group, key))
            keys.add(key)
    if not routes:
        key = os.getenv("GROQ_API_KEY", "").strip()
        if key:
            routes.append(Route("groq", "groq", key))
    if not routes:
        raise ValueError("Set GROQ_API_KEY_1 in .env")
    return routes


async def available_models(client: httpx.AsyncClient, endpoint: Endpoint, route: Route) -> set[str]:
    response = await client.get(f"{endpoint.base_url}/models", headers=route.headers())
    if not response.is_success:
        raise ValueError(f"{route.label} model catalog returned HTTP {response.status_code}")
    return {item["id"] for item in response.json()["data"]}

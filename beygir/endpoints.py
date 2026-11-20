"""API models, credential routing and server-reported limits."""

import os
import re
from dataclasses import dataclass, field

GROQ_URL = "https://api.groq.com/openai/v1"
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
    reasoning: str = "low"
    limits: Limits = Limits()
    max_tokens: int = 4096
    structured_output: bool = True
    token_estimate: int = 2600
    region: str = ""

    def signature(self) -> dict:
        return {"alias": self.alias, "model_id": self.model_id, "api": self.base_url,
                "provider": "bedrock" if self.region else "", "reasoning": self.reasoning,
                "temperature": 0, "max_completion_tokens": self.max_tokens,
                "structured_output": self.structured_output}

@dataclass(frozen=True)
class Route:
    label: str
    quota_group: str
    key: str = field(repr=False)

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"}


def catalog() -> dict[str, Endpoint]:
    region = os.getenv("BEDROCK_REGION", "eu-central-1").strip() or "eu-central-1"
    if not re.fullmatch(r"(?:eu|us|ap|ca|sa|me|af|il|mx)-[a-z]+-\d+", region):
        raise ValueError("BEDROCK_REGION must be an AWS region code")
    models = {
        "gpt20": Endpoint("gpt20", "openai/gpt-oss-20b"),
        "gpt120": Endpoint("gpt120", "openai/gpt-oss-120b"),
        "qwen": Endpoint("qwen", "qwen/qwen3.8-27b", reasoning="none", max_tokens=512, token_estimate=750),
        "allam": Endpoint("allam", "allam-2-7b", reasoning="",
                          limits=Limits(rpd=7000, tpm=6000, tpd=500_000), max_tokens=512, structured_output=False, token_estimate=900),
    }
    bedrock_models = (
        ("aws-oss20", "openai.gpt-oss-20b-1:0", 4096, "low"),
        ("aws-oss120", "openai.gpt-oss-120b-1:0", 4096, "low"),
        ("aws-qwen32", "qwen.qwen3-32b-v1:0", 4096, ""),
        ("aws-glmflash", "zai.glm-4.7-flash", 4096, ""),
        ("aws-glm47", "zai.glm-4.7", 4096, ""),
        ("aws-deepseek", "deepseek.v3.2", 4096, ""),
        ("aws-kimi", "moonshotai.kimi-k2.5", 4096, ""),
        ("aws-nova", "eu.amazon.nova-2-lite-v1:0", 2048, ""),
        ("aws-haiku", "eu.anthropic.claude-haiku-4-5-20251001-v1:0", 2048, ""),
        ("aws-minimax", "minimax.minimax-m2.1", 4096, ""),
    )
    for alias, model, cap, reasoning in bedrock_models:
        models[alias] = Endpoint(
            alias, model, f"https://bedrock-runtime.{region}.amazonaws.com",
            reasoning=reasoning, limits=Limits(rpm=30, rpd=100_000, tpm=100_000, tpd=100_000_000),
            max_tokens=cap, structured_output=False, region=region,
        )
    return models


def configured_routes() -> list[Route]:
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


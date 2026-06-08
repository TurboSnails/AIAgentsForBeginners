#!/usr/bin/env python3
"""
06-python-agent-framework.py
系统消息框架（System Message Framework）— Python 版

本脚本将 Notebook 中的系统提示词生成流程翻译为可直接运行的 Python 脚本。
核心逻辑：使用大语言模型，根据给定的公司、角色和职责，自动生成一份
结构化、详尽的 AI Assistant 系统提示词（System Prompt）。

依赖（二选一）：
    pip install openai python-dotenv              # 推荐，更稳定
    pip install azure-ai-inference python-dotenv  # 备选

身份验证：
    在 .env 文件中配置 GITHUB_TOKEN（或直接从环境变量读取）：
        GITHUB_TOKEN=ghp_xxx  或  GITHUB_TOKEN=github_pat_xxx

常见问题排查：
    1. 若报 "Bad credentials"：
       - 确认 Token 已开启 `models:read` 权限（GitHub → Settings → Developer
         settings → Personal access tokens → 勾选 "Models" 读取权限）
       - Token 过期需重新生成
    2. 若端点不通：脚本默认优先使用 OpenAI SDK 方式（走
       https://models.github.ai/inference），兼容性最好；
       也可手动切换为 Azure AI Inference SDK 方式。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    pass  # 仅用于类型提示

# ── 模型与端点配置 ──────────────────────────────────────
MODEL_NAME = "gpt-4o"

# GitHub Models 端点（OpenAI SDK 方式用）
GITHUB_ENDPOINT = "https://models.github.ai/inference"

# Azure AI Inference 端点（备选）
AZURE_ENDPOINT = "https://models.inference.ai.azure.com"

# ── 示例：Agent 角色定义 ────────────────────────────────
ROLE = "travel agent"
COMPANY = "contoso travel"
RESPONSIBILITY = "booking flights"

# ── 生成器自身的系统提示词 ──────────────────────────────
SYSTEM_PROMPT_FOR_GENERATOR = """\
You are an expert at creating AI agent assistants.
You will be provided a company name, role, responsibilities and other
information that you will use to provide a system prompt for.
To create the system prompt, be descriptive as possible and provide
a structure that a system using an LLM can better understand the role
and responsibilities of the AI assistant.
"""


def load_token() -> str:
    """从 .env 或环境变量读取 GitHub Token。"""
    env_candidates = [
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent.parent.parent.parent / ".env",
    ]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(env_path)
            break

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError(
            "GITHUB_TOKEN 未设置。请在 .env 文件中配置或导出该环境变量。\n"
            "生成地址：https://github.com/settings/tokens"
        )
    return token


# ── 方案 A：OpenAI SDK（推荐，兼容性最好）────────────────

def generate_with_openai_sdk(token: str) -> str:
    """使用 OpenAI SDK 调用 GitHub Models。"""
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "未安装 openai 包，请执行：pip install openai"
        ) from exc

    client = openai.OpenAI(
        base_url=f"{GITHUB_ENDPOINT}/v1",
        api_key=token,
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_FOR_GENERATOR},
            {
                "role": "user",
                "content": (
                    f"You are {ROLE} at {COMPANY} "
                    f"that is responsible for {RESPONSIBILITY}."
                ),
            },
        ],
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0,
    )
    return response.choices[0].message.content


# ── 方案 B：Azure AI Inference SDK（备选）───────────────

def generate_with_azure_sdk(token: str) -> str:
    """使用 Azure AI Inference SDK 调用 GitHub Models。"""
    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.credentials import AzureKeyCredential
    except ImportError as exc:
        raise ImportError(
            "未安装 azure-ai-inference 包，请执行：pip install azure-ai-inference"
        ) from exc

    client = ChatCompletionsClient(
        endpoint=AZURE_ENDPOINT,
        credential=AzureKeyCredential(token),
        # 如果遇到 SSL 证书错误，取消下面一行的注释：
        # connection_verify=False,
    )

    response = client.complete(
        messages=[
            SystemMessage(content=SYSTEM_PROMPT_FOR_GENERATOR),
            UserMessage(
                content=f"You are {ROLE} at {COMPANY} that is responsible for {RESPONSIBILITY}."
            ),
        ],
        model=MODEL_NAME,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0,
    )
    return response.choices[0].message.content


def main() -> None:
    # ── 1. 认证 ───────────────────────────────────────────
    token = load_token()
    print(f"🔑 Token 已加载 ({token[:8]}...)")

    # ── 2. 打印输入信息 ───────────────────────────────────
    print(f"\n📝 正在生成系统提示词 ...")
    print(f"   角色: {ROLE}")
    print(f"   公司: {COMPANY}")
    print(f"   职责: {RESPONSIBILITY}\n")

    # ── 3. 生成系统提示词（优先 OpenAI SDK）───────────────
    try:
        print("🚀 尝试使用 OpenAI SDK 调用 GitHub Models ...")
        system_prompt = generate_with_openai_sdk(token)
        print(system_prompt)
    except Exception as e:
        print(f"⚠️  OpenAI SDK 方式失败: {e}")
        print("🔄 回退到 Azure AI Inference SDK ...")
        system_prompt = generate_with_azure_sdk(token)
        print(system_prompt)


if __name__ == "__main__":
    main()

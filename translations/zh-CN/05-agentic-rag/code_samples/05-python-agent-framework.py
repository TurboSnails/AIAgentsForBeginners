#!/usr/bin/env python3
"""
05-python-agent-framework.py
使用 Microsoft Foundry 的企业级 RAG（Python 版）

此脚本将 .NET 笔记本中的 Agentic RAG 流程翻译为可直接运行的 Python 脚本。
因环境中 azure-ai-projects 2.2.0 不提供 agents 子客户端，
故通过 get_openai_client() 获取 OpenAI 兼容客户端，使用 Assistants API v2 实现。

核心步骤一一对应：
  1. 读取 .env 配置（AZURE_AI_PROJECT_ENDPOINT、AZURE_AI_MODEL_DEPLOYMENT_NAME）
  2. 将本地 document.md 上传到 Azure AI Foundry
  3. 创建 Vector Store 并对文档进行向量化索引
  4. 创建带 file_search 工具的 Persistent Agent
  5. 创建 Thread，发送用户问题，运行 Agent
  6. 打印助手回复并清理资源

依赖：
    pip install azure-ai-projects azure-identity openai python-dotenv

身份验证（三选一）：
    1) 在 .env 中配置 API Key（最简单，推荐）：
       AZURE_AI_API_KEY=你的-api-key
    2) 已安装 Azure CLI 并登录：az login
    3) 配置服务主体环境变量：
       AZURE_CLIENT_ID、AZURE_CLIENT_SECRET、AZURE_TENANT_ID
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import openai
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

if TYPE_CHECKING:
    from openai.types.beta import Assistant, Thread, VectorStore
    from openai.types.beta.threads import Run


def load_config() -> tuple[str, str, str | None]:
    """加载环境变量，返回 (endpoint, model_id, api_key_or_none)。"""
    env_path = (
        Path(__file__).resolve().parent.parent.parent.parent.parent / ".env"
    )
    if env_path.exists():
        load_dotenv(env_path)

    endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    model_id = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
    api_key = os.getenv("AZURE_AI_API_KEY")

    if not endpoint:
        raise ValueError(
            "环境变量 AZURE_AI_PROJECT_ENDPOINT 未设置。"
            "请在 .env 文件中配置或导出该变量。"
        )
    return endpoint, model_id, api_key


def create_openai_client(endpoint: str, api_key: str | None = None):
    """获取已认证的 OpenAI 兼容客户端。

    优先使用 API Key（无需 Azure CLI/登录）：
        在 .env 中配置 AZURE_AI_API_KEY 即可直接运行。

    若无 API Key，则回退到 DefaultAzureCredential：
        1. 环境变量（AZURE_CLIENT_ID / AZURE_CLIENT_SECRET / AZURE_TENANT_ID）
        2. 托管身份（Managed Identity）
        3. Visual Studio Code 登录
        4. Azure CLI 登录（az login）
        5. Azure PowerShell 登录
    """
    if api_key:
        print("🔑 使用 API Key 认证")
        return openai.OpenAI(
            base_url=f"{endpoint.rstrip('/')}/openai/v1",
            api_key=api_key,
        )

    print("🔐 使用 DefaultAzureCredential（需 Azure CLI 或服务主体）")
    project_client = AIProjectClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential(),
    )
    return project_client.get_openai_client()


def upload_document(client, file_path: Path):
    """上传文档并返回文件对象。"""
    print(f"📤 正在上传文档: {file_path.name} ...")
    with open(file_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="assistants")
    print(f"   文件 ID: {uploaded.id}")
    return uploaded


def create_vector_store(client, file_id: str) -> VectorStore:
    """创建 Vector Store 并索引指定文件。"""
    print("🔍 正在创建 Vector Store 并索引文档 ...")
    vector_store = client.vector_stores.create(
        file_ids=[file_id],
        name="python-rag-vectorstore",
    )
    print(f"   Vector Store ID: {vector_store.id}")
    return vector_store


def create_rag_agent(
    client,
    model_id: str,
    vector_store_id: str,
) -> Assistant:
    """创建带 File Search 工具的 Persistent Agent。"""
    instructions = """\
You are an AI assistant designed to answer user questions using only the information retrieved from the provided document(s).

- If a user's question cannot be answered using the retrieved context, **you must clearly respond**:
  "I'm sorry, but the uploaded document does not contain the necessary information to answer that question."
- Do not answer from general knowledge or reasoning. Do not make assumptions or generate hypothetical explanations.
- Do not provide definitions, tutorials, or commentary that is not explicitly grounded in the content of the uploaded file(s).
- If a user asks a question like "What is a Neural Network?", and this is not discussed in the uploaded document, respond as instructed above.
- For questions that do have relevant content in the document (e.g., Contoso's travel insurance coverage), respond accurately, and cite the document explicitly.

You must behave as if you have no external knowledge beyond what is retrieved from the uploaded document.
"""

    agent = client.beta.assistants.create(
        model=model_id,
        name="PythonRAGAgent",
        instructions=instructions,
        tools=[{"type": "file_search"}],
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store_id],
            }
        },
    )
    print(f"🤖 Agent 已创建: {agent.name} (ID: {agent.id})")
    return agent


def ask_question(
    client,
    agent: Assistant,
    question: str,
) -> str:
    """创建 Thread、发送问题、运行 Agent 并返回助手文本。"""
    thread: Thread = client.beta.threads.create()
    print(f"💬 Thread 已创建: {thread.id}")
    print(f"❓ 用户提问: {question}\n")

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=question,
    )

    run: Run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=agent.id,
    )
    print(f"⏱️  Run 状态: {run.status}\n")

    messages = client.beta.threads.messages.list(
        thread_id=thread.id,
        order="desc",
        limit=1,
    )

    for msg in messages:
        if msg.role == "assistant":
            parts = []
            for c in msg.content:
                if hasattr(c, "text") and c.text:
                    parts.append(c.text.value)
            return "\n".join(parts) if parts else "⚠️ 助手回复为空"

    return "⚠️ 未获取到助手回复"


def cleanup(
    client,
    vector_store: VectorStore,
    file,
    agent: Assistant,
) -> None:
    """删除 Vector Store、文件和 Agent（生产环境建议保留）。"""
    print("🧹 正在清理资源 ...")
    client.vector_stores.delete(vector_store.id)
    client.files.delete(file.id)
    client.beta.assistants.delete(agent.id)
    print("✅ 资源已清理")


def main() -> None:
    # ── 1. 配置 ─────────────────────────────────────────────
    endpoint, model_id, api_key = load_config()
    print(f"✅ 端点: {endpoint}")
    print(f"✅ 模型: {model_id}")

    # ── 2. 文档路径 ─────────────────────────────────────────
    script_dir = Path(__file__).resolve().parent
    document_path = script_dir / "document.md"
    if not document_path.exists():
        raise FileNotFoundError(f"演示文档未找到: {document_path}")
    print(f"📄 演示文档: {document_path}")

    # ── 3~11. 执行 RAG 流程 ─────────────────────────────────
    client = create_openai_client(endpoint, api_key)
    print("✅ 已获取 OpenAI 兼容客户端")

    uploaded_file = upload_document(client, document_path)
    vector_store = create_vector_store(client, uploaded_file.id)
    agent = create_rag_agent(client, model_id, vector_store.id)

    answer = ask_question(
        client,
        agent,
        "Can you explain Contoso's travel insurance coverage?",
    )
    print(f"🤖 Agent 回复:\n{answer}\n")

    cleanup(client, vector_store, uploaded_file, agent)


if __name__ == "__main__":
    main()

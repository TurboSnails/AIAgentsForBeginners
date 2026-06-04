"""
Lesson 03 - Agentic Design Patterns (Python Agent Framework)
第03课 - 智能体设计模式（Python Agent Framework）

This script demonstrates three foundational design patterns for building
effective AI agents:
本脚本演示了构建高效 AI 智能体的三个基础设计模式：

1. Clear Agent Instructions        - 清晰的智能体指令
2. Structured Output with Pydantic Models - 使用 Pydantic 模型的结构化输出
3. Single Responsibility Agents    - 单一职责智能体
"""

# ------------------------------------------------------------------------------
# 1. 导入必要的标准库和第三方库
# ------------------------------------------------------------------------------

import asyncio
# asyncio 是 Python 内置的异步 I/O 库，用于编写并发代码。
# 在本脚本中，所有与 AI 智能体的交互都是异步的（网络请求不会阻塞程序），
# 因此需要使用 async/await 语法，并通过 asyncio.run() 启动事件循环。

import logging
# logging 模块用于记录程序运行日志。此处我们用它来抑制来自 Azure 集成的冗长日志，
# 使控制台输出更简洁，便于学习者聚焦于核心逻辑。

import os
# os 模块提供与操作系统交互的接口，常用来读取环境变量。
# 例如：os.getenv("FOUNDRY_PROJECT_ENDPOINT") 用于获取 Azure AI Foundry 的服务端点。

import sys
# sys 模块提供对 Python 运行时环境的访问。我们使用 sys.exit(1) 在缺少必要配置时优雅地终止程序。

from pathlib import Path
# pathlib 是面向对象的文件系统路径处理库，比传统的字符串路径更易读、更安全。
# 这里用于从当前脚本位置向上回溯，定位仓库根目录。

from typing import Annotated
# typing.Annotated 允许为类型添加元数据（metadata）。
# 在定义智能体工具（@tool）时，我们用它为参数添加描述性说明，
# 这些说明会被大模型读取，帮助它正确理解何时、如何调用该工具。

from dotenv import load_dotenv
# python-dotenv 库用于从 .env 文件中加载环境变量。
# 这样可以将敏感信息（如 API 密钥、服务端点）与代码分离，避免硬编码，提升安全性。

from pydantic import BaseModel
# Pydantic 是一个数据验证库。BaseModel 允许我们用类声明数据的结构和类型，
# 并在运行时自动验证。本课用它定义结构化输出模式（Schema），
# 强制大模型返回符合预期的 JSON 格式。

from agent_framework import tool
# 从 Microsoft Agent Framework 导入 tool 装饰器。
# 被 @tool 装饰的 Python 函数可以被大模型识别并调用，实现“工具使用（Tool Use）”。
# approval_mode="never_require" 表示此工具无需人工审批即可执行。

from agent_framework_foundry import FoundryChatClient
# FoundryChatClient 连接 Azure AI Foundry（需 az login + FOUNDRY_* / AZURE_AI_* 环境变量）。

from azure.identity import AzureCliCredential
# 仅在使用 Foundry 时需要；通过 `az login` 获取令牌。

# ------------------------------------------------------------------------------
# 加载 zh-CN 复用的 OpenAI 兼容 Chat Client（独立于框架的 Responses 路径）
# ------------------------------------------------------------------------------
# `agent_framework.openai.OpenAIChatClient` 走的是 OpenAI 新 `/v1/responses` 端点，
# 而 MiniMax 等 OpenAI-compatible 服务只实现了老的 `/v1/chat/completions` 端点。
# 仓库内置的 `translations/zh-CN/.agents/chat_clients/openai_compat.py` 用 httpx
# 直接调老端点，避开这一不兼容问题。该目录以 `.` 开头不能作为 Python 包名导入，
# 因此用 importlib 按文件位置加载。
import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path as _Path

_CHAT_CLIENT_MODULE_NAME = "zhcn_openai_compat"
_chat_client_spec = _importlib_util.spec_from_file_location(
    _CHAT_CLIENT_MODULE_NAME,
    _Path(__file__).resolve().parents[2] / ".agents" / "chat_clients" / "openai_compat.py",
)
_chat_client_module = _importlib_util.module_from_spec(_chat_client_spec)
_sys.modules[_CHAT_CLIENT_MODULE_NAME] = _chat_client_module
_chat_client_spec.loader.exec_module(_chat_client_module)

OpenAICompatChatClient = _chat_client_module.OpenAICompatChatClient
from_env = _chat_client_module.from_env
# 复用别名，保持与原代码风格一致
OpenAIChatClient = OpenAICompatChatClient

# ------------------------------------------------------------------------------
# 2. 日志级别配置：抑制来自 Azure 集成模块的冗余日志
# ------------------------------------------------------------------------------
# 默认情况下，Azure SDK 会输出大量调试/信息日志，干扰学习者的阅读体验。
# 下面两行将 "agent_framework.azure" 和 "agent_framework.foundry" 的日志级别设为 ERROR，
# 即只显示错误级别的日志，隐藏 INFO 和 DEBUG 日志。
logging.getLogger("agent_framework.azure").setLevel(logging.ERROR)
logging.getLogger("agent_framework.foundry").setLevel(logging.ERROR)


# ------------------------------------------------------------------------------
# 3. 辅助函数：定位仓库根目录并加载环境变量
# ------------------------------------------------------------------------------

def _find_repo_root() -> Path:
    """
    从当前脚本所在目录向上逐层查找，直到找到包含 requirements.txt 的目录。
    该目录被视为仓库根目录，从而确保 .env 文件路径正确。
    """
    current = Path(__file__).resolve().parent
    # Path(__file__) 获取当前脚本的相对路径；.resolve() 将其转为绝对路径；.parent 获取所在文件夹。

    for parent in (current, *current.parents):
        # current.parents 生成所有上级目录的序列。遍历当前目录及其所有祖先目录。
        if (parent / "requirements.txt").is_file():
            # 使用 / 运算符拼接路径（pathlib 特性）。is_file() 检查文件是否存在。
            return parent

    # 如果遍历到根目录仍未找到 requirements.txt，抛出异常并给出中文提示。
    raise RuntimeError("找不到仓库根目录（缺少 requirements.txt）")


def _load_env() -> None:
    """
    加载 .env 文件中的环境变量，并进行键名映射。
    课程示例统一使用 AZURE_AI_* 前缀，而 Foundry SDK 读取 FOUNDRY_* 前缀，
    因此需要在此处做兼容性转换，避免学习者手动重命名环境变量。
    """
    load_dotenv(_find_repo_root() / ".env")
    # 先定位仓库根目录，再加载根目录下的 .env 文件。

    # 如果 FOUNDRY_PROJECT_ENDPOINT 未设置，但 AZURE_AI_PROJECT_ENDPOINT 已存在，
    # 则将后者的值同步给前者，确保 FoundryChatClient 能正确读取配置。
    if not os.getenv("FOUNDRY_PROJECT_ENDPOINT") and os.getenv("AZURE_AI_PROJECT_ENDPOINT"):
        os.environ["FOUNDRY_PROJECT_ENDPOINT"] = os.environ["AZURE_AI_PROJECT_ENDPOINT"]

    # 同理，同步模型部署名称：AZURE_AI_MODEL_DEPLOYMENT_NAME -> FOUNDRY_MODEL。
    if not os.getenv("FOUNDRY_MODEL") and os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME"):
        os.environ["FOUNDRY_MODEL"] = os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"]


def _create_chat_client():
    """
    按优先级选择模型提供商（无需 Azure 也可用 MiniMax / GitHub Models）：
    1. MINIMAX_API_KEY  2. GITHUB_TOKEN  3. Azure Foundry  4. OPENAI_API_KEY

    返回的 client 实现了 BaseChatClient 协议（可被 .as_agent() 使用）。
    对 OpenAI 兼容的服务使用本地 `OpenAICompatChatClient`（走老
    /v1/chat/completions 端点），不依赖框架的 Responses 路径。
    """
    if os.getenv("MINIMAX_API_KEY"):
        print("使用 MiniMax 提供商")
        return OpenAICompatChatClient(
            model=os.getenv("MINIMAX_MODEL_ID", "MiniMax-M2.7"),
            base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
            api_key=os.environ["MINIMAX_API_KEY"],
        )

    if os.getenv("GITHUB_TOKEN"):
        print("使用 GitHub Models 提供商")
        return OpenAICompatChatClient(
            model=os.getenv("GITHUB_MODEL_ID", "gpt-4o-mini"),
            base_url=os.getenv(
                "GITHUB_ENDPOINT", "https://models.inference.ai.azure.com"
            ),
            api_key=os.environ["GITHUB_TOKEN"],
        )

    if os.getenv("FOUNDRY_PROJECT_ENDPOINT") and os.getenv("FOUNDRY_MODEL"):
        print("使用 Azure AI Foundry 提供商（需已执行 az login）")
        return FoundryChatClient(credential=AzureCliCredential())

    if os.getenv("OPENAI_API_KEY"):
        print("使用 OpenAI 提供商")
        return OpenAICompatChatClient(
            model=os.getenv("OPENAI_MODEL_ID", "gpt-4o"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ["OPENAI_API_KEY"],
        )

    print("未检测到可用模型配置。请在仓库根目录 .env 中配置以下任一组：")
    print("  MiniMax:     MINIMAX_API_KEY, MINIMAX_BASE_URL, MINIMAX_MODEL_ID")
    print("  GitHub:      GITHUB_TOKEN, GITHUB_ENDPOINT, GITHUB_MODEL_ID")
    print("  Azure:       AZURE_AI_PROJECT_ENDPOINT, AZURE_AI_MODEL_DEPLOYMENT_NAME + az login")
    print("  OpenAI:      OPENAI_API_KEY, OPENAI_MODEL_ID（可选）")
    sys.exit(1)


# ------------------------------------------------------------------------------
# 4. Pydantic 模型：定义结构化输出的数据模式（Schema）
# ------------------------------------------------------------------------------
# 大模型通常返回自然语言文本，但在实际应用中，我们经常需要可预测、可解析的 JSON。
# Pydantic BaseModel 通过声明字段名称和类型，让大模型“知道”应该输出什么格式的数据，
# 并在返回后自动校验和反序列化为 Python 对象。

class DestinationRecommendation(BaseModel):
    """
    单个目的地推荐的数据结构。
    每个字段的类型和名称都向大模型传达了明确的输出要求。
    """
    destination: str
    # destination（目的地）：字符串类型，例如 "Tokyo"。

    available: bool
    # available（是否可预订）：布尔类型，明确告知该目的地当前是否适合前往。

    best_season: str
    # best_season（最佳季节）：字符串类型，例如 "Mar-Apr"（3-4月）。

    highlights: list[str]
    # highlights（亮点）：字符串列表，列出该目的地吸引人的景点或活动。
    # 例如 ["Beach", "Architecture", "Nightlife"]。

    estimated_budget_usd: int
    # estimated_budget_usd（预估预算，美元）：整数类型，给出大致费用，方便用户决策。


class TravelRecommendations(BaseModel):
    """
    一次完整推荐响应的顶层数据结构，包含多个目的地和个性化附言。
    """
    recommendations: list[DestinationRecommendation]
    # recommendations（推荐列表）：由 DestinationRecommendation 对象组成的列表，
    # 允许一次返回多个候选目的地。

    personalized_note: str
    # personalized_note（个性化备注）：字符串类型，大模型可以在这里添加针对用户的额外建议或问候。


# ------------------------------------------------------------------------------
# 5. 工具函数：模拟查询目的地信息
# ------------------------------------------------------------------------------
# @tool 装饰器将普通 Python 函数暴露给大模型，使其在推理过程中可以主动调用。
# approval_mode="never_require" 表示该工具不需要人工审批，适合只读、安全无副作用的操作。

@tool(approval_mode="never_require")
def get_destination_details(destination: Annotated[str, "The destination to look up"]) -> str:
    """
    查询某个度假目的地的详细信息。

    参数说明：
        destination: Annotated[str, "The destination to look up"]
        - destination（目的地）：字符串类型。
        - Annotated 中的第二个参数是对该参数的描述，大模型会据此判断用户提到的是哪个参数。
          例如当用户提到 "Tokyo" 时，大模型知道应将其传入 destination 参数。

    返回值：
        str - 包含该目的地可用性、最佳季节、亮点和预算的字符串。
    """
    details = {
        "Barcelona": "Available. Best: May-Jun. Beach, architecture, nightlife. ~$2000/week",
        "Tokyo": "Available. Best: Mar-Apr. Culture, food, technology. ~$2500/week",
        "Cape Town": "Not available. Best: Nov-Mar. Nature, wine, adventure. ~$1800/week",
    }
    # 这是一个硬编码的模拟数据库，用于演示目的。在生产环境中，这里通常会调用真实的 API 或数据库。

    return details.get(destination, f"{destination}: No information available.")
    # dict.get(key, default) 方法：如果 destination 存在于字典中则返回对应值，
    # 否则返回默认字符串，提示暂无该目的地信息。


# ------------------------------------------------------------------------------
# 6. 主函数：演示三种智能体设计模式
# ------------------------------------------------------------------------------

async def main() -> None:
    """
    脚本主入口。按顺序演示三种设计模式：
    1. 清晰的智能体指令（Clear Agent Instructions）
    2. 结构化输出（Structured Output with Pydantic Models）
    3. 单一职责智能体（Single Responsibility Agents）
    """

    # --- 步骤 A：加载环境变量并校验配置 ---
    _load_env()
    # 调用上面定义的函数，从 .env 文件读取环境变量，并做 AZURE_AI_* -> FOUNDRY_* 的键名转换。

    chat_client = _create_chat_client()
    # 根据 .env 自动选择 MiniMax / GitHub Models / Azure Foundry / OpenAI。

    # ==================================================================
    # 模式 1：清晰的智能体指令（Clear Agent Instructions）
    # ==================================================================
    # 核心思想：提示工程（Prompt Engineering）是构建高效智能体的第一步。
    # 指令越清晰、越具体，大模型的行为就越稳定、越符合预期。
    # 好的指令应包含：角色定义、职责边界、输出要求、语气风格。

    print("=" * 60)
    print("Pattern 1: Clear Agent Instructions")
    print("模式 1：清晰的智能体指令")
    print("=" * 60)

    # 使用 chat_client.as_agent() 创建一个智能体实例。
    # name：智能体的唯一标识名，在日志和多智能体系统中很有用。
    # instructions：系统指令（System Prompt），决定了智能体的行为模式。
    travel_concierge = chat_client.as_agent(
        name="TravelConcierge",
        instructions="""You are a luxury travel concierge named Alex. Your role is to:
1. Understand the traveler's preferences (budget, climate, activities)
2. Check destination availability before making recommendations
3. Provide detailed, personalized travel suggestions
4. Always mention visa requirements and best travel seasons
Be warm, professional, and enthusiastic about travel.""",
    )
    # 指令要点解析：
    # - "luxury travel concierge named Alex"：赋予角色名和身份，增强交互拟人化。
    # - 4 条具体职责：明确告诉模型每一步该做什么，减少随机发挥。
    # - "warm, professional, and enthusiastic"：语气要求，控制输出风格。

    # 调用智能体的 run() 方法，传入用户提问。
    # 这是一个异步操作，因此需要 await。
    # 大模型会基于上面的系统指令生成回答。
    response = await travel_concierge.run(
        "I'd love a week-long vacation somewhere with great food and history. Budget around $2500."
    )
    print(response)
    print()
    # 打印智能体的回复，并空一行，为下一个模式分隔视觉空间。

    # ==================================================================
    # 模式 2：结构化输出（Structured Output with Pydantic Models）
    # ==================================================================
    # 核心思想：当智能体的输出需要被下游程序自动处理（如存入数据库、渲染前端页面）时，
    # 纯文本的不确定性太高。通过 Pydantic 模型定义 JSON Schema，
    # 可以强制大模型返回结构化的、可预测的数据。

    print("=" * 60)
    print("Pattern 2: Structured Output with Pydantic Models")
    print("模式 2：使用 Pydantic 模型的结构化输出")
    print("=" * 60)

    # 创建第二个智能体，这次给它配备了工具（tools=[get_destination_details]），
    # 并在指令中明确要求返回结构化 JSON。
    structured_agent = chat_client.as_agent(
        name="StructuredTravelExpert",
        instructions=(
            "You are a travel expert. Recommend destinations based on traveler preferences. "
            "Use the get_destination_details tool. Return structured JSON matching the schema."
        ),
        tools=[get_destination_details],
        # tools 列表告诉智能体：在回答过程中，如果需要查询目的地信息，可以调用此工具。
    )

    # 在 run() 的 options 参数中传入 {"response_format": TravelRecommendations}，
    # 这样 FoundryChatClient 会自动将 Pydantic 模型转换为 JSON Schema 发送给大模型，
    # 并将返回的 JSON 自动反序列化为 TravelRecommendations 实例。
    response = await structured_agent.run(
        "Recommend 3 destinations for a culture-loving traveler with a $2500 budget",
        options={"response_format": TravelRecommendations},
    )

    if response:
        print(response)
        # 此时 response 不是普通字符串，而是 TravelRecommendations 对象。
        # 我们可以直接访问 response.recommendations[0].destination 等字段。
    print()

    # ==================================================================
    # 模式 3：单一职责智能体（Single Responsibility Agents）
    # ==================================================================
    # 核心思想：就像软件开发中的“单一职责原则（SRP）”一样，
    # 每个智能体只负责一件事，可以显著降低复杂度、提升可维护性和可复用性。
    # 复杂的任务可以通过多个专业智能体协作完成（即多智能体系统 Multi-Agent System 的雏形）。

    print("=" * 60)
    print("Pattern 3: Single Responsibility Agents")
    print("模式 3：单一职责智能体")
    print("=" * 60)

    # --- 智能体 A：目的地研究专家 ---
    # 职责边界非常明确：只评估和推荐目的地，不碰机票、酒店等后勤事务。
    # 这种分工让提示词更聚焦，减少了大模型“越界”输出的概率。
    destination_agent = chat_client.as_agent(
        name="DestinationExpert",
        tools=[get_destination_details],
        instructions="""You are a destination research specialist. Your only job is to:
1. Evaluate destinations based on traveler preferences
2. Check availability using the provided tool
3. Return a short ranked list with pros/cons
Do NOT discuss flights, hotels, or logistics — another agent handles that.""",
    )
    # 最后一句 "Do NOT discuss..." 是负面指令（Negative Instruction），
    # 明确划定禁区，防止模型在响应中混入不相关的信息。

    # --- 智能体 B：行程后勤规划师 ---
    # 职责：根据已选目的地制定详细行程、推荐航班和酒店。
    # 它与 DestinationExpert 形成上下游关系：先定目的地，再做后勤。
    logistics_agent = chat_client.as_agent(
        name="LogisticsPlanner",
        instructions="""You are a travel logistics planner. Your only job is to:
1. Create a day-by-day itinerary for the chosen destination
2. Suggest flight and hotel options within the stated budget
3. Note visa requirements and travel insurance recommendations
Do NOT recommend destinations — another agent handles that.""",
    )

    # --- 第一阶段：由 DestinationExpert 推荐目的地 ---
    dest_response = await destination_agent.run(
        "I want a week of culture and food for under $2500. Where should I go?"
    )
    print("=== Destination Expert ===")
    print("=== 目的地专家 ===")
    print(dest_response)
    print()

    # --- 第二阶段：将第一阶段的结果作为输入，传给 LogisticsPlanner ---
    # 这种“链式调用”是多智能体协作最简单的形式：输出即输入。
    # 在实际生产系统中，可能会使用更复杂的工作流引擎（如 Sequential、Concurrent、
    # Conditional Workflow）来编排多个智能体。
    logistics_response = await logistics_agent.run(
        f"Plan a week-long trip based on this recommendation:\n{dest_response}"
    )
    print("=== Logistics Planner ===")
    print("=== 后勤规划师 ===")
    print(logistics_response)


# ------------------------------------------------------------------------------
# 7. 脚本入口
# ------------------------------------------------------------------------------
# 当直接运行此 Python 文件时（而非被其他文件导入时），__name__ 的值为 "__main__"。
# asyncio.run(main()) 启动事件循环并运行异步主函数。
# 如果尝试 import 此文件，main() 不会自动执行，这是 Python 的最佳实践。

if __name__ == "__main__":
    asyncio.run(main())

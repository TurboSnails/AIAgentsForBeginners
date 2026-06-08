"""
Lesson 04 - Tool Use Design Pattern (Python Agent Framework)
第04课 - 工具使用设计模式（Python Agent Framework）

This script demonstrates the **Tool Use** design pattern for AI agents with the
Microsoft Agent Framework (Python). We cover:

本脚本演示了使用 Microsoft Agent Framework (Python) 为 AI 智能体实现
**工具使用（Tool Use）** 设计模式。涵盖：

1. Defining function tools with the `@tool` decorator and typed parameters
   - 使用 `@tool` 装饰器和类型化参数定义功能工具
2. Letting the model discover each tool's schema automatically
   - 提供工具架构，让模型自动发现每个工具的能力
3. Controlling tool execution with `approval_mode`
   - 通过 `approval_mode` 控制工具执行
4. Returning **structured output** with Pydantic + `response_format`
   - 通过 Pydantic 模型和 `response_format` 返回结构化输出

The scenario is a **travel booking agent** that can query destinations,
check availability, retrieve flight info, and (with approval) book a flight.
场景是一个能够查询目的地、检查可用性、检索航班信息并在审批后完成
航班预订的旅行预订代理。
"""

# ------------------------------------------------------------------------------
# 1. 导入必要的标准库和第三方库
# ------------------------------------------------------------------------------

import asyncio
# asyncio 是 Python 内置的异步 I/O 库。
# 智能体的运行是异步操作（网络请求不会阻塞），因此需要 async/await 语法，
# 并通过 asyncio.run() 启动事件循环。

import logging
# logging 用于调节日志级别：抑制来自 Azure 集成的冗长日志，
# 让控制台输出聚焦核心逻辑，便于学习者阅读。

import os
# os 模块用于读取环境变量（如 API Key、端点等敏感配置）。

import sys
# sys 模块用于在缺少必要配置时优雅地终止程序（sys.exit(1)）。

from pathlib import Path
# pathlib 是面向对象的文件系统路径处理库。
# 这里用它从当前脚本位置向上回溯，定位仓库根目录，从而加载 .env。

from typing import Annotated
# Annotated 允许为类型添加元数据（描述）。
# 在 @tool 装饰的函数中，我们用它为参数添加说明，
# 这些说明会被大模型读取，帮助它理解每个参数的含义。

from dotenv import load_dotenv
# python-dotenv 从 .env 文件中加载环境变量，避免在代码中硬编码敏感信息。

from pydantic import BaseModel
# Pydantic 的 BaseModel 用于声明结构化输出的字段和类型。
# 配合 response_format 可强制大模型返回符合 Schema 的 JSON。

from agent_framework import tool
# 从 Microsoft Agent Framework 导入 tool 装饰器。
# 被 @tool 装饰的 Python 函数会被注册为智能体可调用的工具。

from agent_framework_foundry import FoundryChatClient
# FoundryChatClient 连接 Azure AI Foundry（需 az login + FOUNDRY_* / AZURE_AI_* 环境变量）。

from azure.identity import AzureCliCredential
# AzureCliCredential 通过 `az login` 获取令牌，仅在选用 Foundry 时需要。

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
    for parent in (current, *current.parents):
        if (parent / "requirements.txt").is_file():
            return parent
    raise RuntimeError("找不到仓库根目录（缺少 requirements.txt）")


def _load_env() -> None:
    """
    加载 .env 文件中的环境变量，并做必要的键名映射。
    课程示例统一使用 AZURE_AI_* 前缀，而 Foundry SDK 读取 FOUNDRY_* 前缀，
    因此需要在此处做兼容性转换，避免学习者手动重命名环境变量。
    """
    load_dotenv(_find_repo_root() / ".env")

    if not os.getenv("FOUNDRY_PROJECT_ENDPOINT") and os.getenv("AZURE_AI_PROJECT_ENDPOINT"):
        os.environ["FOUNDRY_PROJECT_ENDPOINT"] = os.environ["AZURE_AI_PROJECT_ENDPOINT"]

    if not os.getenv("FOUNDRY_MODEL") and os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME"):
        os.environ["FOUNDRY_MODEL"] = os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"]


def _create_chat_client():
    """
    按优先级选择模型提供商（无需 Azure 也可用 MiniMax / GitHub Models）：
    1. MINIMAX_API_KEY  2. GITHUB_TOKEN  3. Azure Foundry  4. OPENAI_API_KEY

    返回的 client 实现了 BaseChatClient 协议，可被 .as_agent() 使用。
    对 OpenAI 兼容的服务使用本地 OpenAICompatChatClient（走老的
    /v1/chat/completions 端点），不依赖框架的 Responses 路径。

    注意事项：
    - 与 Lesson 03 不同，这里在 Azure 路径下额外返回 AzureCliCredential，
      因为后续工具需要走 Microsoft Foundry 注册（如有）。如果你的环境只
      使用 MiniMax / GitHub Models / OpenAI，可忽略 Foundry 路径。
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
# 4. Pydantic 模型：结构化输出 Schema
# ------------------------------------------------------------------------------
# 大模型通常返回自然语言文本，但当结果需要被下游程序（前端、数据库、流水
# 线）消费时，必须是可预测的 JSON。Pydantic 模型正是这种契约：把"期望的
# 字段名 + 类型"告诉模型，并把返回值反序列化为 Python 对象。

class FlightDetails(BaseModel):
    """单条航班的结构化信息。把这些字段拆开能强制模型按字段填空，
    避免在 flight_details 里夹带英文散文描述（"direct flight, duration ~2h15min"）
    等无法被映射表翻译的内容。
    """
    flight_number: str
    # 航班号，如 "BA 2042"（行业标识符，保留英文/数字原样）

    origin_airport: str
    # 出发地机场 IATA 代码，如 "LHR"

    destination_airport: str
    # 目的地机场 IATA 代码，如 "BCN"

    departure_time: str
    # 起飞时间，如 "08:30"

    arrival_time: str
    # 抵达时间，如 "11:45"

    price_usd: int
    # 价格（美元）


class BookingRecommendation(BaseModel):
    """单条目的地预订推荐的数据结构。"""
    destination: str
    # destination: 字符串，目的地中文名（模型按 schema 填，脚本兜底翻译）

    available: bool
    # available: 布尔值，表示该目的地当前是否可订

    flight: FlightDetails
    # flight: 结构化航班信息（航班号/起降机场/时间/价格）

    estimated_cost: int
    # estimated_cost: 整数，估算总费用（美元）

    note: str = ""
    # note: 可选备注（如"无可达航班"），模型留空时脚本兜底填中文


class TravelPlan(BaseModel):
    """一次完整推荐响应的顶层结构，包含多条候选目的地。"""
    recommendations: list[BookingRecommendation]


# ------------------------------------------------------------------------------
# 5. 工具函数：模拟旅行预订的查询接口
# ------------------------------------------------------------------------------
# @tool 装饰器把一个普通 Python 函数变成智能体可调用的工具。
#   - 文档字符串 (docstring)    -> 工具描述，模型据此判断"何时调用"
#   - 类型注解 + Annotated 描述 -> 参数模式，模型据此判断"如何传参"
#   - approval_mode             -> 是否需要在执行前得到人工批准
#
# approval_mode="never_require"  工具自动执行，适合只读、安全无副作用的操作
# approval_mode="always_require" 每次执行前都需要人工批准，适合副作用操作

@tool(approval_mode="never_require")
def get_destinations() -> list[str]:
    """获取可选的度假目的地列表。"""
    return ["Barcelona", "Paris", "Berlin", "Tokyo", "Sydney", "New York City"]


@tool(approval_mode="never_require")
def check_availability(
    destination: Annotated[str, "要查询的目的地"],
) -> str:
    """查询某目的地的预订可用性。"""
    availability = {
        "Barcelona": "可预订 - 剩余 3 个名额",
        "Paris": "可预订",
        "Berlin": "已售罄",
        "Tokyo": "可预订 - 剩余 1 个名额",
        "Sydney": "可预订",
        "New York City": "可预订",
    }
    return availability.get(destination, "未知目的地")


@tool(approval_mode="never_require")
def get_flight_info(
    origin: Annotated[str, "出发地机场代码 (IATA)"],
    destination: Annotated[str, "目的地机场代码 (IATA)"],
) -> str:
    """查询两个城市之间的航班信息。"""
    flights = {
        "LHR-BCN": "BA 2042, 08:30 起飞, 11:45 抵达, 350 美元",
        "LHR-CDG": "AF 1081, 09:15 起飞, 11:30 抵达, 280 美元",
        "LHR-NRT": "JL 044, 11:00 起飞, 次日 07:00 抵达, 890 美元",
    }
    return flights.get(
        f"{origin}-{destination}",
        f"从 {origin} 到 {destination} 没有直达航班",
    )


@tool(approval_mode="always_require")
def book_flight(
    origin: Annotated[str, "出发地机场代码 (IATA)"],
    destination: Annotated[str, "目的地机场代码 (IATA)"],
    passenger_name: Annotated[str, "乘客全名"],
) -> str:
    """为乘客预订航班。执行前需要人工审批。"""
    return (
        f"已为乘客 {passenger_name} 预订从 {origin} 到 {destination} 的航班。"
        f"确认号 #TRV-2024-{hash(passenger_name) % 10000:04d}"
    )


# ------------------------------------------------------------------------------
# 6. 主函数：按顺序演示工具使用的各个要点
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# 6. Demo 单元：每个 demo 独立成函数，main 只负责调度
# ------------------------------------------------------------------------------
# 把每一节 demo 封装成独立的 async 函数后，main 只剩"建 client + 顺序
# await"两件事。新增/跳过某节时只动这一个调度块，不再嵌套 if/else 写
# 几十行。返回值/异常全部收敛到各 demo 内部，main 不感知细节。


def _print_travel_plan(plan: "TravelPlan", header: str) -> None:
    """统一打印 TravelPlan 的辅助函数（适配结构化 FlightDetails）。"""
    print(header)
    for rec in plan.recommendations:
        f = rec.flight
        if rec.available:
            flight_str = (
                f"{f.flight_number} {f.origin_airport}→{f.destination_airport} "
                f"{f.departure_time}-{f.arrival_time} ${f.price_usd}"
            )
        else:
            flight_str = rec.note or "无可用航班"
        note = f" | note={rec.note}" if rec.note else ""
        print(
            f"  - {rec.destination} | available={rec.available} | "
            f"flight={flight_str}{note} | cost=${rec.estimated_cost}"
        )


def _looks_like_json_object(text: str) -> bool:
    """粗判：字符串首尾是否像单个 JSON 对象。"""
    if not text:
        return False
    stripped = text.strip()
    return stripped.startswith("{") and stripped.rstrip().endswith("}")


def _extract_structured_tool_args(response) -> str | None:
    """在 response 的 messages 中寻找最后一条 function_call 的 JSON 参数。

    当 provider 不支持 `response_format` 时，OpenAICompatChatClient 会把
    schema 塞进虚拟 tool `submit_structured_output`，模型的最终 JSON 就
    出现在该 tool_call 的 arguments 字段里。函数会扫描 response 里所有
    messages（包括 user/assistant/tool result），返回第一条看起来像合法
    JSON 对象的 arguments 字符串；找不到返回 None。
    """
    import json as _json

    messages = getattr(response, "messages", None) or []
    for msg in reversed(messages):
        contents = getattr(msg, "contents", None) or []
        for c in contents:
            ctype = getattr(c, "type", None)
            name = getattr(c, "name", None)
            if ctype == "function_call" and name == "submit_structured_output":
                args = getattr(c, "arguments", None)
                if isinstance(args, str):
                    try:
                        _json.loads(args)
                        return args
                    except _json.JSONDecodeError:
                        continue
                if isinstance(args, dict):
                    return _json.dumps(args, ensure_ascii=False)
    return None


# ------------------------------------------------------------------------------
# 5.1 英中映射表 + 兜底翻译器
# ------------------------------------------------------------------------------
# Prompt 只能"建议"模型用中文；模型有时仍会从 tool 返回或自有词库里夹英文。
# 这一层在 JSON 解析成功后强制把英文地名/备注转成中文，作为硬保证。
# 航班号（如 BA 2042）和机场代码（如 LHR、BCN）由 schema 强约束保留英文。

_DESTINATION_ZH = {
    "Barcelona": "巴塞罗那",
    "Paris": "巴黎",
    "Berlin": "柏林",
    "Tokyo": "东京",
    "Sydney": "悉尼",
    "New York City": "纽约",
    "London": "伦敦",
    "London Heathrow": "伦敦希思罗",
    "Heathrow": "希思罗",
    "Spain": "西班牙",
    "France": "法国",
    "Germany": "德国",
}

# 无法订到航班时的兜底中文备注
_NO_FLIGHT_NOTE_ZH = "无可用直达航班"


def _zh_ratio(text: str) -> float:
    """计算字符串中 CJK 汉字（含全角符号）的占比。"""
    if not text:
        return 0.0
    zh_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return zh_chars / len(text)


def _localize_plan(plan: "TravelPlan") -> "TravelPlan":
    """对解析后的 TravelPlan 做一次英中规范化。

    schema 已经把航班拆成结构化字段，模型无法夹英文散文；note 字段是
    自由文本兜不住，所以再加一道"中文字符比例"硬约束：note 里中文比例
    不足 50% 时直接覆盖为"暂无中文备注"，保证打印输出永远是中文。
    """
    for rec in plan.recommendations:
        # destination 兜底翻译（schema 没强制字段值语言，模型可能填英文）
        if rec.destination in _DESTINATION_ZH:
            rec.destination = _DESTINATION_ZH[rec.destination]
        # note 字段：先做地名替换，再做中文比例检查；不达标就覆盖
        if not rec.note:
            rec.note = _NO_FLIGHT_NOTE_ZH
        else:
            for en, zh in _DESTINATION_ZH.items():
                if en in rec.note:
                    rec.note = rec.note.replace(en, zh)
            if _zh_ratio(rec.note) < 0.5:
                rec.note = "暂无中文备注"
    return plan

def _try_parse_travel_plan(response, raw_text: str) -> "TravelPlan | None":
    """解析 TravelPlan：先看 response.text，再回退到虚拟 tool 参数。

    返回解析后的 TravelPlan；解析失败返回 None，由调用方决定如何降级
    打印告警。
    """
    candidate = raw_text
    if not _looks_like_json_object(candidate):
        candidate = _extract_structured_tool_args(response) or candidate
    try:
        return TravelPlan.model_validate_json(candidate)
    except Exception:
        return None


async def demo_section1_tool_definitions() -> None:
    """第1节：展示 @tool 装饰器定义的工具元信息（name / approval_mode）。"""
    print("=" * 60)
    print("Section 1: Tools defined with @tool")
    print("第1节：使用 @tool 定义工具")
    print("=" * 60)
    for t in (get_destinations, check_availability, get_flight_info, book_flight):
        print(f"- {t.name}  approval_mode={t.approval_mode}")
    print()


async def demo_section2_multi_tool_agent(chat_client) -> None:
    """第2节：把多个工具挂到同一个 agent，演示模型自主选工具。"""
    print("=" * 60)
    print("Section 2: Agent with multiple tools")
    print("第2节：组合多个工具")
    print("=" * 60)

    travel_agent = chat_client.as_agent(
        name="TravelToolAgent",
        instructions=(
            "你是一名旅行代理。请使用可用工具回答关于目的地、可用性和航班的问题。"
            "所有回复必须使用中文。城市名、地名都用中文表达（如\"巴塞罗那\"、"
            "\"巴黎\"、\"伦敦希思罗\"），但机场代码（如 LHR、BCN）和航班号"
            "（如 BA 2042）保持原样。"
        ),
        tools=[get_destinations, check_availability, get_flight_info],
    )

    response = await travel_agent.run(
        "你们有哪些目的地？哪些还有可用名额？"
    )
    print(response)
    print()


async def demo_section3_structured_output(chat_client) -> None:
    """第3节：tools + response_format，单阶段结构化输出（兼容虚拟 tool 路径）。"""
    print("=" * 60)
    print("Section 3: Structured output from tool-using agent")
    print("第3节：使用工具的结构化输出")
    print("=" * 60)

    # 提示工程要点：当 `response_format` 与工具调用同时存在时，模型（尤其
    # Qwen 系）容易"先讲一段自然语言再走工具"，最后忘了把结论收成 JSON。
    # 指令里要明确：
    #   1) 先调用工具拿数据；
    #   2) **只**输出符合 schema 的 JSON，禁止任何解释/前缀/后缀文本；
    #   3) 给出最少 1 条推荐（避免空列表触发 Pydantic 边界问题）。
    structured_agent = chat_client.as_agent(
        name="StructuredTravelAgent",
        instructions=(
            "你是一名旅行代理。请严格遵守以下规则：\n"
            "1. 首先调用可用工具收集目的地、可用性和航班信息；\n"
            "2. 之后**只**输出符合 schema 的 JSON 对象，禁止任何叙述、"
            "   解释、markdown 围栏或前后缀；\n"
            "3. 至少返回 1 条推荐。\n"
            "4. 按 schema 字段严格填写：\n"
            "   - destination：使用中文城市名（如\"巴塞罗那\"、\"巴黎\"）\n"
            "   - flight.flight_number：航班号原样（如\"BA 2042\"）\n"
            "   - flight.origin_airport / destination_airport：IATA 代码"
            "（如\"LHR\"、\"BCN\"），保持大写\n"
            "   - flight.departure_time / arrival_time：24 小时制"
            "（如\"08:30\"、\"11:45\"）\n"
            "   - flight.price_usd：整数美元价格\n"
            "   - note：中文备注；无可用航班时写\"无可用直达航班\""
        ),
        tools=[get_destinations, check_availability, get_flight_info],
    )

    response = await structured_agent.run(
        "我想从伦敦希思罗机场飞往欧洲某个温暖的城市，帮我看看有哪些可用选项。",
        options={"response_format": TravelPlan},
    )

    if not response:
        return

    # 重要：不要触碰 `response.value` —— 它是 @property，访问即触发
    # Pydantic 校验，模型一旦没返回合规 JSON 就会把整个 asyncio.run
    # 拖崩。在 `tools + response_format` 组合下，Qwen 系模型常会先
    # 讲一段自然语言再调工具，最后忘了把结论收成 JSON。这是模型
    # 行为问题，客户端层没法兜，必须在脚本里做防御。
    #
    # 正确做法：直接拿 response.text，自己调 Pydantic 解析；失败则
    # 降级打印原始文本，保留可观察性，而不是炸进程。
    raw_text = response.text or ""
    print("--- raw model output (response_format=TravelPlan) ---")
    print(raw_text)
    print("--- end raw ---")

    # 兼容路径：MiniMax 等不支持 response_format 的提供商会让客户端把
    # schema 塞进虚拟 tool `submit_structured_output`，模型以 tool_call
    # 形式返回 JSON。这种情况下 response.text 是空或自然语言尾巴，
    # 真正的 JSON 在最后一条 function_call.arguments 里。
    plan = _try_parse_travel_plan(response, raw_text)
    if plan is not None:
        plan = _localize_plan(plan)
        _print_travel_plan(plan, "\nParsed TravelPlan:")
    else:
        print(
            "\n[WARN] 模型未返回符合 TravelPlan 的 JSON。"
            "这通常是模型在 tools + response_format 组合下没遵守 schema 约束；"
            "可换用支持 json_schema strict 模式的模型，或拆成『先 tool 拿数据、"
            "再单独一次 run 拿结构化输出』两阶段。"
        )
    print()


async def demo_section3_1_two_phase(chat_client) -> None:
    """第3.1节：两阶段写法（兼容所有 OpenAI-compatible 模型）。

    当模型不严格遵守 json_schema 时，把"拿数据"和"产出结构化结果"拆成
    两次 run，第二次没有 tools，模型更容易乖乖吐 JSON。这种模式在
    Lesson 05+ 的多智能体 / RAG 场景里会被反复用到。
    """
    print("-" * 60)
    print("Section 3.1: Two-phase pattern (robust across providers)")
    print("第3.1节：两阶段写法（跨提供商稳健）")
    print("-" * 60)

    # 阶段 1：拿数据，自然语言总结
    gather_agent = chat_client.as_agent(
        name="TravelGatherer",
        instructions=(
            "你是一名旅行助理。请使用工具查找目的地、检查可用性并获取航班信息，"
            "用简短的自然语言进行总结即可，本阶段无需输出 JSON。"
            "所有回复必须使用中文，城市名/地名用中文表达（如\"巴塞罗那\"、"
            "\"伦敦希思罗\"），机场代码（如 LHR、BCN）和航班号（如 BA 2042）"
            "保持原样。"
        ),
        tools=[get_destinations, check_availability, get_flight_info],
    )
    gathered = await gather_agent.run(
        "我想从伦敦希思罗机场飞往欧洲某个温暖的城市，帮我看看有哪些可用选项。"
    )
    gathered_text = gathered.text if gathered else ""
    print("Phase 1 summary:")
    print(gathered_text)
    print()

    # 阶段 2：不带 tools，强制 JSON
    formatter_agent = chat_client.as_agent(
        name="TravelFormatter",
        instructions=(
            "你负责把旅行摘要转成符合 schema 的严格 JSON。"
            "**只**输出 JSON 对象，禁止任何叙述或 markdown 围栏，至少返回 1 条推荐。"
            "按 schema 字段严格填写：\n"
            "  - destination：使用中文城市名（如\"巴塞罗那\"、\"巴黎\"）\n"
            "  - flight.flight_number：航班号原样（如\"BA 2042\"）\n"
            "  - flight.origin_airport / destination_airport：IATA 代码"
            "（如\"LHR\"、\"BCN\"），保持大写\n"
            "  - flight.departure_time / arrival_time：24 小时制"
            "（如\"08:30\"、\"11:45\"）\n"
            "  - flight.price_usd：整数美元价格\n"
            "  - note：中文备注；无可用航班时写\"无可用直达航班\""
        ),
    )
    print("Phase 2 summary:")
    formatted = await formatter_agent.run(
        "请把以下旅行摘要转换成 TravelPlan 格式的 JSON：\n"
        f"{gathered_text}",
        options={"response_format": TravelPlan},
    )
    if formatted:
        raw_text2 = formatted.text or ""
        print(f"|||-->{raw_text2}")
        print("|||------------------------>")
        plan2 = _try_parse_travel_plan(formatted, raw_text2)
        if plan2 is not None:
            plan2 = _localize_plan(plan2)
            _print_travel_plan(plan2, "Phase 2 parsed TravelPlan:")
        else:
            print(
                f"[WARN] 第二阶段仍未得到合规 JSON\n"
                f"raw: {raw_text2[:400]}"
            )
    print()


async def demo_section4_approval_mode(chat_client) -> None:
    """第4节：approval_mode=always_require 触发人工审批。

    关键流程（agent_framework 1.x 的审批协议）：
        1. 第一次 agent.run()：模型返回 function_call 调 book_flight。
        2. FunctionInvocationLayer 检测到该工具 approval_mode ==
           "always_require"，**不执行**它，而是把每个 function_call 包成
           Content(type="function_approval_request", ...) 放在
           response.messages 里返回。
        3. 调用方从 response.messages 里扫出这些 request 字段，提示用户
           y/n；用户批准后用 Content.to_function_approval_response(
           approved=True) 把它转成 response，把所有 response 拼成一条
           新的 user/assistant 消息喂回 agent.run()。
        4. 第二次 run：模型看到 function_result 里有"已批准"标识，
           框架执行 book_flight，工具结果作为 function_result 回流，
           模型据此生成最终回复。
    """
    print("=" * 60)
    print("Section 4: Tool approval mode")
    print("第4节：工具审批模式")
    print("=" * 60)
    print("Tool name:", book_flight.name)
    print("Approval mode:", book_flight.approval_mode)
    print()

    # 演示：把 book_flight 加入工具集。智能体在真实运行时会先请求审批。
    approval_demo_agent = chat_client.as_agent(
        name="BookingAgent",
        instructions=(
            "你是一名旅行预订助理。请在用户提供出发地、目的地和乘客姓名后"
            "使用工具预订航班；book_flight 工具需要审批。"
            "所有回复必须使用中文。"
        ),
        tools=[check_availability, get_flight_info, book_flight],
    )

    # 关键：approval 走"两轮 run"协议，
    # 第 1 轮模型返回 function_call
    # 后框架拦截、把它包成 function_approval_request 留在 assistant消息里；
    # 第 2 轮 framework 又需要把那条 assistant 消息 + 用户的
    # approval_response 一起送给 chat client，才能让服务端在同一个
    # tool_call_id 上对应起 function_call/function_result。
    # 显式开一个 AgentSession 让 framework 自动维护这段历史。
    from agent_framework import AgentSession
    session = AgentSession()

    # ---- 第 1 轮：等模型发出审批请求 ----
    response = await approval_demo_agent.run(
        "请帮我为乘客 Alice Smith 预订一班从 LHR 到 BCN 的航班。",
        session=session,
    )

    # 扫出所有 function_approval_request
    from agent_framework import Message
    import json as _json_approval

    approval_requests = [
        c
        for m in (response.messages or [])
        for c in (m.contents or [])
        if getattr(c, "type", None) == "function_approval_request"
    ]
    if not approval_requests:
        # 走到这里说明模型没调工具，审批路径没触发
        print(response)
        print("[INFO] 模型未触发审批（可能没调 book_flight），跳过第二轮。")
        return

    print("模型请求执行以下需要审批的工具：")
    for c in approval_requests:
        fc = c.function_call
        print(f"  - {fc.name}({_json_approval.dumps(fc.arguments, ensure_ascii=False)})")

    # 把所有 request 转成 response（全部批准；要"全部拒绝"传 False）
    approval_responses = [
        c.to_function_approval_response(approved=True)
        for c in approval_requests
    ]

    # ---- 第 2 轮：把"批准"内容喂回 agent.run() ----
    # 同一 session 让 framework 自动把第 1 轮的 assistant(approval_request)
    # 重新塞回 prepped_messages；再把 approval_response 作为新的 input。
    # framework 内部的 _replace_approval_contents_with_results 会就地
    # 把 approval_request 还原成 function_call、把 approval_response
    # 替换成 function_result（msg.role="tool"），再调 chat client。
    follow_up_msg = Message(role="user", contents=approval_responses)
    final = await approval_demo_agent.run(follow_up_msg, session=session)
    print(final)
    print()


async def demo_summary() -> None:
    """结尾：四节要点回顾。"""
    print("=" * 60)
    print("Summary")
    print("总结")
    print("=" * 60)
    print(
        "1. 使用 @tool 装饰器 + 类型注解定义工具；\n"
        "2. 通过 tools=[...] 把多个工具提供给智能体；\n"
        "3. 通过 response_format=PydanticModel 返回结构化输出；\n"
        "4. 通过 approval_mode 控制工具是否需要人工审批。"
    )


async def main() -> None:
    """
    主入口：依次演示：
    1. 使用 @tool 装饰器定义工具
    2. 创建一个带有多个工具的智能体
    3. 工具调用的结构化输出（Pydantic + response_format）
    4. 工具的审批模式（approval_mode）
    """
    _load_env()
    chat_client = _create_chat_client()

    # 顺序与 1/2/3/3.1/4 一致；要跳过某节，注释掉对应行即可
    # await demo_section1_tool_definitions()
    # await demo_section2_multi_tool_agent(chat_client)
    # await demo_section3_structured_output(chat_client)
    # await demo_section3_1_two_phase(chat_client)
    await demo_section4_approval_mode(chat_client)
    # await demo_summary()


# ------------------------------------------------------------------------------
# 7. 脚本入口
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())

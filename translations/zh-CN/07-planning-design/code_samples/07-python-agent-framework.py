"""
Lesson 07 - 规划设计模式（Python Agent Framework）

本脚本演示了使用微软代理框架的 **规划设计模式** 用于 AI 代理。
你将学习如何将复杂的旅行请求拆分为结构化的子任务，
将它们分配给专门的代理，并执行生成的计划——这一切都使用由 Pydantic 模型驱动的结构化输出完成。

要点：
1. 任务分解 — 前台规划代理使用 Pydantic 模型将复杂请求拆分为结构化子任务
2. 结构化输出 — 通过 response_format 让代理返回经过验证的 TravelPlan 对象
3. 计划执行 — 礼宾代理使用专家工具迭代子任务，执行计划并报告结果
"""

# ------------------------------------------------------------------------------
# 1. 导入必要的标准库和第三方库
# ------------------------------------------------------------------------------

import asyncio
import logging
import os
from typing import Annotated

from pydantic import BaseModel
from agent_framework import tool
from agent_framework.azure import AzureAIProjectAgentProvider
from azure.identity import AzureCliCredential

# ------------------------------------------------------------------------------
# 2. 日志级别配置：抑制来自 Azure 集成模块的冗余日志
# ------------------------------------------------------------------------------
logging.getLogger("agent_framework.azure").setLevel(logging.ERROR)


# ------------------------------------------------------------------------------
# 3. Pydantic 模型：结构化输出 Schema
# ------------------------------------------------------------------------------
# 大模型通常返回自然语言文本，但当结果需要被下游程序消费时，
# 必须是可预测的 JSON。Pydantic 模型正是这种契约。

class TravelSubTask(BaseModel):
    """旅行计划中的单个子任务。"""
    task_id: int
    description: str
    assigned_agent: str  # "flight_agent", "hotel_agent", "activity_agent"
    priority: str  # "high", "medium", "low"
    dependencies: list[int] = []


class TravelPlan(BaseModel):
    """完整旅行计划的顶层结构。"""
    destination: str
    trip_duration_days: int
    subtasks: list[TravelSubTask]
    total_estimated_budget_usd: int
    notes: str


# ------------------------------------------------------------------------------
# 4. 工具函数：模拟旅行预订的专用接口
# ------------------------------------------------------------------------------
# @tool 装饰器把普通 Python 函数变成智能体可调用的工具。
#   - 文档字符串 (docstring)    -> 工具描述，模型据此判断"何时调用"
#   - 类型注解 + Annotated 描述 -> 参数模式，模型据此判断"如何传参"

@tool
def book_flight(
    destination: Annotated[str, "The destination city"],
    departure_date: Annotated[str, "Departure date (YYYY-MM-DD)"],
    return_date: Annotated[str, "Return date (YYYY-MM-DD)"],
) -> str:
    """Search and book flights for the trip."""
    return (
        f"Flight booked to {destination}: {departure_date} → {return_date}, "
        f"confirmation #FLT-{hash(destination) % 10000:04d}"
    )


@tool
def reserve_hotel(
    city: Annotated[str, "The city for the hotel"],
    check_in: Annotated[str, "Check-in date (YYYY-MM-DD)"],
    check_out: Annotated[str, "Check-out date (YYYY-MM-DD)"],
    guests: Annotated[int, "Number of guests"],
) -> str:
    """Reserve a hotel room in the destination city."""
    return (
        f"Hotel reserved in {city}: {check_in} to {check_out} for {guests} guests, "
        f"confirmation #HTL-{hash(city) % 10000:04d}"
    )


@tool
def book_activity(
    activity_name: Annotated[str, "Name of the activity or tour"],
    date: Annotated[str, "Date of the activity (YYYY-MM-DD)"],
    participants: Annotated[int, "Number of participants"],
) -> str:
    """Book a tour, museum visit, or other activity."""
    return (
        f"Activity booked: {activity_name} on {date} for {participants} people, "
        f"confirmation #ACT-{hash(activity_name) % 10000:04d}"
    )


# ------------------------------------------------------------------------------
# 5. 主函数：演示规划设计模式的完整流程
# ------------------------------------------------------------------------------

async def main() -> None:
    """
    主入口：依次演示规划设计模式的三个环节：
    1. 前台规划代理分解任务并生成结构化 TravelPlan
    2. 打印生成的计划
    3. 礼宾代理按依赖顺序执行子任务
    """

    # --------------------------------------------------------------------------
    # 5.1 初始化 Azure AI Project Agent Provider
    # --------------------------------------------------------------------------
    # AzureAIProjectAgentProvider 需要 AZURE_AI_PROJECT_ENDPOINT 环境变量，
    # 并通过 AzureCliCredential 使用 `az login` 获取身份令牌。
    provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())

    # --------------------------------------------------------------------------
    # 5.2 创建规划代理（前台协调员）
    # --------------------------------------------------------------------------
    # 规划代理根据高级旅行请求生成结构化的 TravelPlan，
    # 将请求分解为子任务，设定优先级，并识别依赖关系。
    planning_agent = await provider.create_agent(
        name="TravelPlanner",
        instructions=(
            "You are a travel planning agent. When given a travel request:\n"
            "1. Break it into specific subtasks (flights, hotels, activities, logistics)\n"
            "2. Assign each subtask to the appropriate specialist agent\n"
            "3. Set priorities and identify dependencies between tasks\n"
            "4. Estimate the total budget"
        ),
    )

    response = await planning_agent.run(
        "Plan a 7-day trip to Paris for a couple interested in art, cuisine, and history. "
        "Budget around $5000.",
        options={"response_format": TravelPlan},
    )

    # 兼容处理：某些路径下 run() 直接返回解析后的对象，
    # 有些路径下返回 response wrapper，需取 .value
    plan = response
    if hasattr(response, "value") and response.value is not None:
        plan = response.value

    if plan is None:
        print("[WARN] 规划代理未返回有效结果。")
        return

    # --------------------------------------------------------------------------
    # 5.3 打印生成的结构化旅行计划
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("Generated Travel Plan")
    print("生成的旅行计划")
    print("=" * 60)
    print(f"Destination:  {plan.destination}")
    print(f"Duration:     {plan.trip_duration_days} days")
    print(f"Budget:       ${plan.total_estimated_budget_usd}")
    if hasattr(plan, "notes") and plan.notes:
        print(f"Notes:        {plan.notes}")
    print(f"\nSubtasks:")
    for task in plan.subtasks:
        deps = f" (deps: {task.dependencies})" if task.dependencies else ""
        print(
            f"  [{task.priority}] {task.task_id}. {task.description} "
            f"→ {task.assigned_agent}{deps}"
        )
    print()

    # --------------------------------------------------------------------------
    # 5.4 创建礼宾代理并使用专用工具执行计划
    # --------------------------------------------------------------------------
    # 礼宾代理按照依赖顺序遍历计划中的子任务，
    # 并将每个子任务分派给相应的工具。
    concierge_agent = await provider.create_agent(
        name="Concierge",
        instructions=(
            "You are a travel concierge executing a structured travel plan.\n"
            "Use the available tools to fulfil each subtask. Work through the subtasks in order, "
            "respecting dependencies. Summarise the results when finished."
        ),
        tools=[book_flight, reserve_hotel, book_activity],
    )

    subtask_lines = "\n".join(
        f"- [{t.priority}] {t.task_id}. {t.description} "
        f"(agent: {t.assigned_agent}, deps: {t.dependencies})"
        for t in plan.subtasks
    )
    execution_prompt = (
        f"Execute the following travel plan for {plan.destination} "
        f"({plan.trip_duration_days} days, ${plan.total_estimated_budget_usd} budget):\n"
        f"{subtask_lines}"
    )

    print("=" * 60)
    print("Concierge Execution")
    print("礼宾代理执行")
    print("=" * 60)
    exec_response = await concierge_agent.run(execution_prompt)
    print(exec_response)
    print()

    # --------------------------------------------------------------------------
    # 5.5 摘要
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("Summary")
    print("总结")
    print("=" * 60)
    print(
        "1. 任务分解 — 前台规划代理将复杂请求拆分为结构化子任务，\n"
        "   分配给具有优先级和依赖关系的专家代理。\n"
        "2. 结构化输出 — 通过 response_format 让代理返回经过验证的\n"
        "   TravelPlan 对象，使下游处理更加可靠。\n"
        "3. 计划执行 — 礼宾代理使用专家工具迭代子任务，执行计划并报告结果。\n"
        "\n此模式将'做什么'（规划）与'如何做'（执行）分离，\n"
        "使代理更加模块化、可测试且更易扩展。"
    )


# ------------------------------------------------------------------------------
# 6. 脚本入口
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())

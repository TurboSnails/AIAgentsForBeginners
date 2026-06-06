"""
OpenAI 兼容 Chat Client —— 针对只实现老式 /v1/chat/completions 端点的提供商
（例如 MiniMax）而设计。

【背景】
    微软 Agent Framework 自带的 `OpenAIChatClient` 走的是 OpenAI 较新的
    `/v1/responses` 端点，但绝大多数第三方 OpenAI 兼容服务（MiniMax、
    各类国内代理、Azure OpenAI 的部分部署）至今仍只实现老的
    `/v1/chat/completions` 端点。本模块用 httpx 手写一个针对老端点的
    `BaseChatClient` 子类，使得
        chat_client.as_agent(...)
    这套上层代码可以无缝替换底层 chat 协议。

【一次完整往返的流程（以 Lesson 04 跑 MiniMax 为例）】
    travel_agent.run(user_msg)
       │
       ▼ （框架 Agent 类）
    Agent.run()
       │   准备 session_messages / chat_options
       │   处理 middleware / history / compaction
       ▼
    Agent._call_chat_client(ctx, stream=False)
       │   return self.client.get_response(...)
       ▼
    self.client.get_response(messages, stream=False, options=...)
       │   return self._inner_get_response(...)
       ▼
    OpenAICompatChatClient._inner_get_response(messages, stream, options)
       │   我们重写的钩子：拒绝 stream，转发到 _get_response
       ▼
    OpenAICompatChatClient._get_response(messages, options)
       │   ① validate options
       │   ② build payload（messages → tools → response_format）
       │   ③ POST /chat/completions（带重试）
       │   ④ parse response（text / tool_calls → Content）
       ▼
    返回 ChatResponse

【导出符号】
    OpenAICompatChatClient: 替代 `OpenAIChatClient` 的客户端，协议兼容
                            `BaseChatClient`。
    from_env:              工厂方法，从 `<PROVIDER>_API_KEY/_BASE_URL/
                            _MODEL_ID` 环境变量直接拼出一个 client。

【环境变量（api_key 必填，其余可选）】
    <PROVIDER>_API_KEY          API key，例如 `MINIMAX_API_KEY`
    <PROVIDER>_BASE_URL         含 scheme + host + /v1，例如
                                `https://api.minimaxi.com/v1`
    <PROVIDER>_MODEL_ID         模型名，例如 `MiniMax-M3`
    <PROVIDER>_SUPPORTS_RESPONSE_FORMAT   可选；设为 "0"/"false"/"no"
                                强制走"虚拟 tool"路径（结构化输出），
                                设为 "1"/"true"/"yes" 强制走原生
                                `response_format`，留空则按 base_url
                                自动探测。

【结构化输出（response_format）行为差异】
    • OpenAI / Azure / GitHub Models：原生支持 `response_format=
      PydanticModel`（自动转 strict json_schema），用 `_convert_
      response_format` 直接发原生字段。
    • MiniMax：忽略 `response_format`，改走"虚拟 tool"路径 —— 把
      schema 内联到名为 `submit_structured_output` 的 tool 的
      `parameters` 里，强制 `tool_choice` 指向它，再在 system
      里告诉模型必须以该 tool 提交 JSON。这是 MiniMax 官方在文档
      里推荐的"结构化输出"姿势。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Awaitable, Mapping, Sequence

import httpx

# --------------------------------------------------------------------------
# 与 Agent Framework 的接口契约
# --------------------------------------------------------------------------
# BaseChatClient: 抽象客户端基类，子类实现 `_inner_get_response` / 流式
# ChatResponse:   框架侧的"一次模型响应"封装
# Message / Content: 框架侧对 system/user/assistant/tool 等消息的统一表示
# FunctionInvocationLayer: 框架自带的"工具调用层"，混进来后 .run() 会
#                          自动把 tool_calls 翻译成 function_call 类型的
#                          Content，再把工具结果回填成 function_result。
from agent_framework import BaseChatClient, ChatResponse, Message
from agent_framework._types import Content
from agent_framework._tools import FunctionInvocationLayer

# --------------------------------------------------------------------------
# 模块级 logger：所有诊断/重试日志统一走这里，方便上层调 logging 调节。
# --------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class OpenAICompatChatClient(FunctionInvocationLayer, BaseChatClient):
    """老式 `/v1/chat/completions` 协议的最小完整实现。

    支持：
        · 非流式 chat completion
        · system / user / assistant / tool 四种角色消息
        · OpenAI 风格的 `tools` / `tool_choice` function calling
        · Pydantic → JSON-Schema 结构化输出
          （原生 `response_format` + 虚拟 tool 两种姿势）

    不支持：
        · 流式（streaming）—— 主动抛 `NotImplementedError`，因为仓库所有
          课程 notebook 都没用到流式，少一个分支更安全。

    多重继承顺序 `FunctionInvocationLayer, BaseChatClient` 不可调换：
    `FunctionInvocationLayer` 负责在 `.run()` 内部把 assistant 返回的
    `tool_calls` 翻译成 `Content(type="function_call", …)` 并递归调用
    工具，缺了它就退化成纯文本客户端。
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 180.0,
        max_retries: int = 3,
        supports_response_format: bool | None = None,
    ) -> None:
        """初始化客户端。

        参数：
            model:        服务端模型名，如 "MiniMax-M3"
            api_key:      Bearer token
            base_url:     服务根 URL（含 scheme/host/可选 /v1），需能被
                          拼接上 `/chat/completions` 形成完整端点
            timeout:      单次 HTTP 请求超时秒数
            max_retries:  网络层重试次数（仅在超时/断连/协议错误时重试，
                          4xx/5xx 不重试）
            supports_response_format: 显式声明该提供商是否支持原生
                          `response_format`。None 表示按 base_url 自动
                          探测（推荐）。
        """
        super().__init__()
        self.model = model
        self._api_key = api_key
        # 去掉尾部斜杠，确保 `base_url + "/chat/completions"` 拼出来合法
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        # 0 禁用重试；只对瞬时网络错误重试，绝不重试 4xx（鉴权/参数错）
        self._max_retries = max(0, max_retries)

        # ------------------------------------------------------------------
        # 结构化输出能力探测
        # ------------------------------------------------------------------
        # MiniMax 等服务静默丢弃 `response_format`，导致
        # `options={"response_format": PydanticModel}` 变成空操作，
        # 智能体始终吐自然语言。`_detect_response_format_support` 按
        # base_url 决定走哪条路径；用户也可通过
        # `<PROVIDER>_SUPPORTS_RESPONSE_FORMAT` 环境变量或构造参数强制覆盖。
        if supports_response_format is None:
            env_flag = os.getenv(
                _sup_env_var(self._base_url, self.model), ""
            ).strip().lower()
            if env_flag in ("0", "false", "no"):
                supports_response_format = False
            elif env_flag in ("1", "true", "yes"):
                supports_response_format = True
            else:
                supports_response_format = _detect_response_format_support(
                    self._base_url
                )
        self._supports_response_format = supports_response_format

    @property
    def service_url(self) -> str:
        # BaseChatClient 抽象属性：返回 base_url 即可。
        return self._base_url

    # ------------------------------------------------------------------ #
    # BaseChatClient 契约实现
    # ------------------------------------------------------------------ #
    # 上层 agent.run() 最终会调到这里。本类只暴露非流式入口，流式场景
    # 直接抛 NotImplementedError，让调用方早失败。
    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse]:
        """框架入口：收到 messages + options，返回一个 awaitable。

        流程：
            stream=True  → 抛 NotImplementedError（仓库未用）
            stream=False → 立即把协程对象交给 _get_response 执行
        """
        if stream:
            raise NotImplementedError(
                "OpenAICompatChatClient does not implement streaming. "
                "Use non-streaming agent runs (default)."
            )
        return self._get_response(messages, options)

    async def _get_response(
        self,
        messages: Sequence[Message],
        options: Mapping[str, Any],
    ) -> ChatResponse:
        """非流式响应主流程（最关键的一处）。

        完整步骤：
            ① 验证并规范化 options
            ② 把 agent_framework 的 Message 列表 + tools/response_format
               拼成 OpenAI 协议要求的 dict payload
            ③ 用 httpx 发 POST 请求；网络层瞬时错误做指数退避重试
            ④ 服务端 4xx/5xx → 把响应体原样抛出去，便于排查
            ⑤ 把 OpenAI 响应 dict 解析成 ChatResponse
        """
        # ---------- ① options 规范化 ----------
        validated = await self._validate_options(options)

        # ---------- ② 构造 payload ----------
        payload = self._build_payload(messages, validated)

        # ---------- ③ / ④ 发请求 + 重试 ----------
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # 仅在网络层瞬时错误（超时、断连、协议错误）下重试；
        # 收到 4xx/5xx 直接跳出循环，由下方统一报错 —— 重试没意义。
        attempts = self._max_retries + 1
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(1, attempts + 1):
                try:
                    resp = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    break  # 拿到响应就跳出（哪怕是 4xx），不再重试
                except (
                    httpx.TimeoutException,
                    httpx.NetworkError,
                    httpx.RemoteProtocolError,
                ) as exc:
                    # 已经用完所有重试次数 → 抛汇总异常
                    if attempt >= attempts:
                        raise RuntimeError(
                            f"Chat completions request failed after {attempts} "
                            f"attempt(s) (model={self.model}): {exc}"
                        ) from exc
                    # 指数退避，封顶 8s
                    backoff = min(2 ** (attempt - 1), 8)
                    logger.warning(
                        "Chat completions transient error (attempt %d/%d): %s. "
                        "Retrying in %ds.",
                        attempt,
                        attempts,
                        exc,
                        backoff,
                    )
                    await asyncio.sleep(backoff)

        # 4xx / 5xx：把服务端返回的 error body 原样带上抛出去，
        # 远比一个泛化的 exception 容易排查。
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Chat completions request failed "
                f"(status={resp.status_code}, model={self.model}): "
                f"{resp.text[:500]}"
            )

        # ---------- ⑤ 解析响应 ----------
        data = resp.json()
        return self._parse_response(data)

    # ------------------------------------------------------------------ #
    # Payload 构造：Message + options → OpenAI 协议 dict
    # ------------------------------------------------------------------ #
    def _build_payload(
        self,
        messages: Sequence[Message],
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        """把上层传进来的 messages + options 翻译成 OpenAI chat 协议。

        关键产出：
            {
                "model": ...,
                "messages": [
                    {"role": "system",   "content": "..."},
                    {"role": "user",     "content": "..."},
                    {"role": "assistant","content": "...", "tool_calls": [...]},
                    {"role": "tool",     "tool_call_id": "...", "content": "..."},
                    ...
                ],
                "tools":         [...],      # 可选
                "tool_choice":   "auto"/{...}, # 可选
                "response_format": {...},     # 可选（仅原生路径）
                "temperature":   ...,
                "max_tokens":    ...,
                ...
            }
        """
        # ---------- ① 转换 messages ----------
        chat_messages: list[dict[str, Any]] = []
        for m in messages:
            # 1 个 framework Message 可能展开成 1~N 个 OpenAI 消息
            # （比如 assistant 既含 text 又含 tool_calls）
            chat_messages.extend(self._convert_message(m))

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
        }

        # ---------- ② 透传温度/采样参数 ----------
        # ChatOptions 把 temperature / top_p / max_tokens / stop / seed
        # 这些都放到 options 里；逐个拷过来，None 跳过。
        for key in ("temperature", "top_p", "max_tokens", "stop", "seed"):
            value = options.get(key)
            if value is not None:
                payload[key] = value

        # ---------- ③ 注册工具 ----------
        # tools 是 agent_framework 的 AITool / FunctionTool 列表，每个都有
        # name / description / parameters。_resolve_tool_parameters 会把
        # 不同版本的"参数 schema"统一成 dict。
        if tools := options.get("tools"):
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": _resolve_tool_parameters(t),
                    },
                }
                for t in tools
            ]
            # "auto" 表示模型自决要不要调、调哪个
            payload["tool_choice"] = "auto"

        # ---------- ④ 结构化输出（response_format） ----------
        # response_format 既是上层传过来的 Pydantic 类 / 字典 / None，
        # 也是 OpenAI 协议里的"输出约束"字段。这里按提供商能力分两路：
        #   · 支持 → 原生 response_format
        #   · 不支持 → 注入虚拟 tool
        response_format = options.get("response_format")
        if response_format is not None:
            if self._supports_response_format:
                # 原生：发出 {"type":"json_schema","json_schema":{...strict}}
                payload["response_format"] = self._convert_response_format(
                    response_format
                )
            else:
                # 虚拟 tool 路径：把 schema 当成"提交 JSON"的 tool 的参数，
                # 强 pin tool_choice 指向它，并塞一条 system 指令。
                # 这是 MiniMax 文档推荐的"结构化输出"姿势。
                schema = self._schema_for_response_format(response_format)
                if schema is not None:
                    virtual_name = "submit_structured_output"
                    payload.setdefault("tools", []).append(
                        {
                            "type": "function",
                            "function": {
                                "name": virtual_name,
                                "description": (
                                    "Submit your final answer as a JSON "
                                    "object that matches this schema. Call "
                                    "this tool exactly once with the result."
                                ),
                                "parameters": schema,
                            },
                        }
                    )
                    # 强 pin：模型这一轮必须调这个 tool
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": virtual_name},
                    }
                    # 补一条 system 指令，避免模型忽略 tool_choice。
                    # 若开头已有 system 消息则追加到其 content，
                    # 否则插入新条目，避免产生两条 system 消息。
                    _instruction = (
                        "When the user's request requires a "
                        "structured response, you MUST call the "
                        "`submit_structured_output` tool with a "
                        "JSON object that matches its schema. Do "
                        "NOT respond with free-form text or "
                        "markdown in that case."
                    )
                    if chat_messages and chat_messages[0].get("role") == "system":
                        chat_messages[0]["content"] = (
                            chat_messages[0]["content"] + "\n\n" + _instruction
                        )
                    else:
                        chat_messages.insert(
                            0, {"role": "system", "content": _instruction}
                        )
                    logger.info(
                        "Injected virtual tool `submit_structured_output` "
                        "with inline schema (no $ref) for structured output."
                    )

        return payload

    def _convert_message(self, message: Message) -> list[dict[str, Any]]:
        """把一个框架侧 `Message` 翻译成 1~N 个 OpenAI 消息 dict。

        输入示例（framework 侧）：
            Message(role="assistant", contents=[
                Content(type="text", text="I'll check..."),
                Content(type="function_call", name="get_destinations",
                        call_id="call_1", arguments={}),
            ])

        输出示例（OpenAI 侧）：
            [
                {"role": "assistant", "content": "I'll check...",
                 "tool_calls": [
                     {"id": "call_1", "type": "function",
                      "function": {"name": "get_destinations",
                                   "arguments": "{}"}},
                 ]},
            ]

        关键点：
            · 一条 assistant 消息若同时含 text + tool_calls，二者合并到
              同一个 dict（OpenAI 协议要求）。
            · tool_result 单独成条（role="tool"），不回塞到 assistant 上。
        """
        role = message.role
        out: list[dict[str, Any]] = []

        # 三个桶：纯文本片段、function_call 列表、function_result 列表
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for c in message.contents or []:
            if not isinstance(c, Content):
                # 偶尔有层会把原始字符串塞进来，做一次防御性转换
                text_parts.append(str(c))
                continue
            ctype = c.type
            if ctype == "text" and c.text:
                # 普通文本片段（assistant 思考、user 输入、system 指令等）
                text_parts.append(c.text)
            elif ctype == "function_call" and c.name:
                # 模型决定要调某个工具 —— 把它挂到 assistant 消息上
                args = c.arguments
                if isinstance(args, Mapping):
                    args_str = json.dumps(args, ensure_ascii=False)
                else:
                    # 已经是字符串就原样用；None/空则兜底 "{}"
                    args_str = args or "{}"
                tool_calls.append(
                    {
                        "id": c.call_id or "",
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": args_str,
                        },
                    }
                )
            elif ctype == "function_result":
                # 工具执行结果 —— OpenAI 协议要求 role="tool" 且必须指明
                # 属于哪个 tool_call_id
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": c.call_id or "",
                        "content": c.result
                        if isinstance(c.result, str)
                        else json.dumps(c.result or "", ensure_ascii=False),
                    }
                )
            # 其他 content 类型（图片、文件、hosted vector 等）本客户端
            # 主动忽略 —— 课程 notebook 只用到文本 + 工具调用。

        # ----- 把三个桶组装成 OpenAI 消息 -----

        # 1) tool_result 单独成条，且在 assistant 消息**之前**插入
        if tool_results:
            out.extend(tool_results)

        # 2) assistant + tool_calls
        if tool_calls:
            out.append({
                "role": "assistant",
                "content": "".join(text_parts) or None,
                "tool_calls": tool_calls,
            })
        # 3) 纯文本 assistant
        elif role == "assistant" and text_parts:
            out.append({"role": "assistant", "content": "".join(text_parts)})
        # 4) 纯文本 user / system / developer
        elif text_parts:
            out.append({"role": self._map_role(role), "content": "".join(text_parts)})

        return out

    @staticmethod
    def _map_role(role: str) -> str:
        """把 framework 角色名映射到 OpenAI 角色名。

        `developer` 是 OpenAI 较新的别名（system 的替代品），多数第三方
        服务只认 `system`，做一次归一化兜底。
        """
        if role in ("system", "user", "assistant", "tool", "developer"):
            return "system" if role == "developer" else role
        return "user"

    @staticmethod
    def _convert_response_format(response_format: Any) -> dict[str, Any]:
        """原生路径：把 Pydantic 类 → OpenAI `response_format` 协议对象。

        返回值结构（OpenAI 严格 JSON-Schema 模式）：
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "TravelPlan",
                    "schema": { ...Pydantic 输出的 JSON Schema... },
                    "strict": true,
                },
            }

        同时支持：
            · Pydantic 类 → 用 model_json_schema() 生成 schema
            · 已经是 dict/Mapping 的 → 原样透传
            · 兜底 → 退化成 `{"type": "json_object"}`（自由 JSON）
        """
        if isinstance(response_format, type) and hasattr(
            response_format, "model_json_schema"
        ):
            schema = response_format.model_json_schema()
            # Pydantic v2 默认产 $ref / $defs；MiniMax 等不做服务端解析，
            # 必须在发出去之前就地展开（见 _inline_json_schema_refs）
            schema = _inline_json_schema_refs(schema)
            # OpenAI strict 模式要求每个 object 节点都有
            # additionalProperties=false（防额外字段），做一次递归兜底
            schema = _enforce_additional_properties_false(schema)
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": schema,
                    "strict": True,
                },
            }
        if isinstance(response_format, Mapping):
            # 已是 OpenAI 协议 dict，原样透传
            return dict(response_format)
        # 兜底：让模型吐"自由 JSON"（不绑具体 schema）
        return {"type": "json_object"}

    @staticmethod
    def _schema_for_response_format(response_format: Any) -> dict[str, Any] | None:
        """虚拟 tool 路径：从 Pydantic 类抽出纯净的 JSON Schema dict。

        注意：
            · 只支持 Pydantic 类；Mapping 类型不在虚拟 tool 路径里处理
              （因为不知道给虚拟 tool 起什么名）
            · 必须内联 `$ref`，否则 MiniMax 看到的"参数 schema"是空对象
            · 必须打上 `additionalProperties:false`，让模型知道不能多塞
              字段
        """
        if isinstance(response_format, type) and hasattr(
            response_format, "model_json_schema"
        ):
            schema = response_format.model_json_schema()
            schema = _inline_json_schema_refs(schema)
            schema = _enforce_additional_properties_false(schema)
            return schema
        return None

    # ------------------------------------------------------------------ #
    # 响应解析：OpenAI chat completion dict → ChatResponse
    # ------------------------------------------------------------------ #
    def _parse_response(self, data: dict[str, Any]) -> ChatResponse:
        """把 OpenAI 服务的响应 dict 翻成框架侧的 ChatResponse。

        OpenAI 响应核心结构：
            {
                "id": "chatcmpl-...",
                "model": "MiniMax-M3",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop" | "tool_calls" | "length",
                        "message": {
                            "role": "assistant",
                            "content": "I'll help you...",         // 可空
                            "tool_calls": [                        // 可空
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_destinations",
                                        "arguments": "{}"
                                    }
                                },
                                ...
                            ]
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 123,
                    "completion_tokens": 45,
                    "total_tokens": 168
                }
            }

        我们只读 `choices[0]`，把它的 message 拆成若干个 Content：
            · text     → Content(type="text", text=...)
            · tool_call → Content(type="function_call",
                                  call_id=..., name=..., arguments=...)
        然后装进一个 assistant Message，再包成 ChatResponse 返回。
        上层 agent.run() 拿到这个 ChatResponse 之后，
        FunctionInvocationLayer 会负责把 function_call 翻译成对真实
        Python 工具的调用，调用结果再被回填成 function_result Content，
        由本类在下一次 _get_response 里转成 role="tool" 消息。
        """
        choices = data.get("choices") or []
        if not choices:
            # 没有任何候选 → 视为协议错误，抛出去
            raise RuntimeError(
                f"Provider returned no choices: {json.dumps(data)[:500]}"
            )
        choice = choices[0]
        message = choice.get("message") or {}
        finish = choice.get("finish_reason")

        contents: list[Content] = []
        if text := message.get("content"):
            # 普通 assistant 文本（含 thinking 块的纯文本尾巴）
            contents.append(Content(type="text", text=text))
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments") or "{}"
            try:
                # arguments 是 JSON 字符串 → 解码成 dict/list
                args: Any = json.loads(args_raw)
            except json.JSONDecodeError:
                # 解析失败就保留原始字符串，让上层工具自己处理
                args = args_raw
            contents.append(
                Content(
                    type="function_call",
                    call_id=tc.get("id"),
                    name=fn.get("name"),
                    arguments=args,
                )
            )

        assistant_msg = Message(
            role="assistant",
            contents=contents,
            message_id=data.get("id"),
        )

        # usage 是可选的（部分老版 MiniMax 不返回），用 getattr 兜底
        usage = None
        if u := data.get("usage"):
            from agent_framework._types import UsageDetails

            usage = UsageDetails(
                input_token_count=u.get("prompt_tokens"),
                output_token_count=u.get("completion_tokens"),
                total_token_count=u.get("total_tokens"),
            )

        return ChatResponse(
            messages=[assistant_msg],
            response_id=data.get("id"),
            model=data.get("model") or self.model,
            finish_reason=finish,
            usage_details=usage,
            raw_representation=data,  # 把原始 dict 一并保留，便于调试
        )


def _resolve_tool_parameters(tool: Any) -> dict[str, Any]:
    """把 agent_framework 的工具对象归一化成 JSON-Schema dict。

    为什么需要这个函数？
        agent_framework 1.x 里 `FunctionTool.parameters` 是一个**方法**
        （按需生成 schema），旧版里是 property，某些用户代码又会直接传
        一个 dict 进来。这里三种情况都要 handle，不能粗暴 `.parameters`
        否则可能拿到一个 bound-method 对象（不可 JSON 序列化）。

    返回：合法的 JSON Schema dict（如 `{"type":"object","properties":...}`）
    """
    params = getattr(tool, "parameters", None)
    if callable(params):
        # 1.x：method；调用一下拿 schema
        params = params()
    if params is None:
        return {"type": "object", "properties": {}}
    if isinstance(params, Mapping):
        return dict(params)
    # 最后一搏：尝试从底层函数生成 Pydantic 风格 schema
    fn = getattr(tool, "func", None) or getattr(tool, "function", None)
    if fn is not None and hasattr(fn, "model_json_schema"):
        return fn.model_json_schema()
    return {"type": "object", "properties": {}}


# ---------------------------------------------------------------------- #
# Schema 加工工具（OpenAI 严格模式相关）
# ---------------------------------------------------------------------- #

def _enforce_additional_properties_false(node: Any) -> Any:
    """递归地把每个 object 节点打上 `additionalProperties: false`。

    为什么需要？
        OpenAI 严格 JSON-Schema 模式拒绝任何"允许额外字段"的节点。
        Pydantic v2 默认会给顶层 object 加上这个标记，但嵌套的 `$defs`
        / array items 仍可能漏掉。一次防御性递归能省掉一整类 400 错误。
    """
    if isinstance(node, dict):
        if node.get("type") == "object":
            # setdefault 不会覆盖已显式设置的值
            node.setdefault("additionalProperties", False)
        for v in node.values():
            _enforce_additional_properties_false(v)
    elif isinstance(node, list):
        for v in node:
            _enforce_additional_properties_false(v)
    return node


def _inline_json_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """把 Pydantic v2 schema 里的所有 `$ref` 全部就地内联展开。

    为什么需要？
        Pydantic v2 的 `model_json_schema()` 习惯把嵌套模型提到顶层
        `$defs`，然后用 `$ref: "#/$defs/Foo"` 引用。MiniMax 等大多数
        OpenAI 兼容服务**不会**在服务端做 ref 解析，结果是：
            {"items": {"$ref": "#/$defs/BookingRecommendation"}}
        被它们当成"items 是个空对象"，于是模型不肯调虚拟 tool。

    实现要点：
        1. 抽出顶层 `$defs` 表
        2. 遍历剩余树，遇到 `$ref: "#/$defs/Foo"` 就用对应定义的
           **深拷贝**替换，并继续向下解析（递归内联）
        3. 维护一个 `expanding` 集合防循环引用 —— Pydantic 实际不会
           生成循环 schema，这只是安全网

    返回：完全 self-contained、不含 `$ref`/`$defs` 的 schema dict。
    """
    schema = dict(schema)  # 浅拷贝，避免原地 mutate 调用方传入的对象
    defs: dict[str, Any] = schema.pop("$defs", {})
    expanding: set[str] = set()

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str):
                ref = node["$ref"]
                if ref.startswith("#/$defs/"):
                    name = ref.split("/")[-1]
                    if name in defs and name not in expanding:
                        # 标记正在展开，避免死循环
                        expanding.add(name)
                        try:
                            return _resolve(defs[name])
                        finally:
                            expanding.discard(name)
                # 未知 / 外部 ref，保持原样
                return node
            # 普通的 dict：每个 value 都递归
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(schema)


# ---------------------------------------------------------------------- #
# 便利工厂：从环境变量拼出一个 client
# ---------------------------------------------------------------------- #

def from_env(prefix: str) -> OpenAICompatChatClient:
    """读 `<PREFIX>_API_KEY/_BASE_URL/_MODEL_ID` 直接拼客户端。

    用法：
        from translations.zh_CN.agents.chat_clients.openai_compat import from_env
        client = from_env("MINIMAX")   # → MINIMAX_API_KEY / MINIMAX_BASE_URL / ...

    还会顺便读 `<PREFIX>_SUPPORTS_RESPONSE_FORMAT`：
        · "0"/"false"/"no"  → 强制走虚拟 tool 路径
        · "1"/"true"/"yes"  → 强制走原生 response_format
        · 缺省 / 其它值    → 按 base_url 自动探测
    """
    api_key = os.environ[f"{prefix}_API_KEY"]
    base_url = os.environ.get(
        f"{prefix}_BASE_URL", "https://api.minimax.io/v1"
    )
    model = os.environ.get(f"{prefix}_MODEL_ID", "MiniMax-M2.7")
    supports_response_format: bool | None = None
    env_flag = os.getenv(f"{prefix}_SUPPORTS_RESPONSE_FORMAT", "").strip().lower()
    if env_flag in ("0", "false", "no"):
        supports_response_format = False
    elif env_flag in ("1", "true", "yes"):
        supports_response_format = True
    return OpenAICompatChatClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        supports_response_format=supports_response_format,
    )


# ---------------------------------------------------------------------- #
# 提供商能力探测
# ---------------------------------------------------------------------- #

# 已知"OpenAI 兼容 /v1/chat/completions 端点会静默忽略 response_format"
# 的服务商。MiniMax 官方文档明确说 response_format 仅在原生 API 且仅
# 对 MiniMax-Text-01 提供 —— 这就是为什么这些 host 要走虚拟 tool 路径。
# 后续若发现其它服务也行为相同，扩展这个集合即可。
_NO_RESPONSE_FORMAT_HOSTS: frozenset[str] = frozenset(
    {
        "api.minimax.io",
        "api.minimaxi.com",
    }
)


def _detect_response_format_support(base_url: str) -> bool:
    """按 base_url 判断该提供商是否真的支持 `response_format`。

    启发式策略：
        · 不在已知黑名单里的 host 一律认为支持（覆盖 OpenAI、Azure、
          GitHub Models 等绝大多数正常实现）
        · 黑名单里"宁杀错不放过"：False 顶多让结构化输出走虚拟 tool
          路径，不会破坏功能正确性
    """
    host = base_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0].lower()
    return host not in _NO_RESPONSE_FORMAT_HOSTS


def _sup_env_var(base_url: str, model: str) -> str:
    """生成"per-host"的 response_format 覆盖环境变量名。

    客户端在构造时不知道用户用了什么 provider 前缀，因此用 host 名
    反推一个稳定的环境变量名。如果自动探测猜错，用户可显式 export
    这个变量来覆盖。
    """
    host = base_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0].lower()
    return f"OPENAI_COMPAT_DISABLE_RESPONSE_FORMAT_{host.replace('.', '_').upper()}"

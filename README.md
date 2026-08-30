# Coding Agent

一个从零实现的轻量编程智能体。它使用 OpenAI 兼容的 Chat Completions 原生 tool
calling，在用户指定的本地工作区内读写文件、精确修改文本并执行命令。项目不依赖任何
Agent 框架，也不使用服务端托管的代码执行或文件工具。

## 运行流程

核心循环位于 `src/coding_agent/agent.py`：

1. 创建 system prompt 和用户任务消息；
2. 将完整消息历史与本地工具定义发送给模型；
3. 解析一轮返回的零个、一个或多个 tool calls；
4. 在本地逐个分发工具，并将每个结构化结果按 `tool_call_id` 回传；
5. 继续下一轮，直到模型返回最终文本、请求失败或达到最大步数。

工具失败不会让进程无提示崩溃。未知工具、非法 JSON、路径越界、文件不存在、命令启动
失败、非零退出码和超时都会变成可读的 tool result，让模型有机会修正操作。终端进度只
显示步骤、工具名和成功状态，不打印工具参数或 API Key。

## 项目结构

```text
src/coding_agent/
├── agent.py       # 消息历史、system prompt、多轮循环和终止条件
├── cli.py         # 命令行参数、交互输入和进度输出
├── config.py      # 环境变量与运行参数校验
├── history.py     # 保持完整工具调用轮次的消息历史截断
├── llm.py         # 模型协议和 OpenAI 兼容适配器
├── sessions.py    # 工作区隔离的本地持久化会话
└── tools.py       # 工作区边界、文件工具、命令工具和分发器
tests/             # 全部使用 fake/mock 模型，不访问真实 API
```

`CodingAgent` 只依赖 `ChatModel` 协议和标准化的 `ModelResponse`，并不知道 OpenAI
客户端的存在。更换另一家兼容服务只需调整环境变量；扩展非兼容厂商时，实现同一
`ChatModel.complete(messages, tools)` 协议即可，无需修改 Agent 循环或工具层。

## 环境要求与安装

- Python 3.11 或更高版本
- 一个支持 Chat Completions `tools` / `tool_calls` 的 OpenAI 兼容接口

建议在虚拟环境中安装：

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

macOS / Linux 激活命令为 `source .venv/bin/activate`。
参与开发或运行测试时改用 `python -m pip install -e ".[dev]"` 安装检查工具。

项目只从以下环境变量读取模型配置，不会自动读取 `.env` 文件：

```powershell
$env:LLM_API_KEY="<your-api-key>"
$env:LLM_BASE_URL="https://<provider-host>/v1"
$env:LLM_MODEL="<provider-model-name>"
```

`.env.example` 仅列出变量名和占位值，可以作为配置清单；不要把真实凭据写入仓库。
不同兼容服务的 URL 路径可能不同，请以服务商文档为准。

## 使用方法

将工作区和自然语言任务作为参数传入：

```powershell
coding-agent --workspace C:\path\to\project "修复解析器的空输入错误并运行测试"
```

省略任务时会在终端中交互读取：

```powershell
coding-agent --workspace C:\path\to\project
Task: 为现有 API 增加分页参数和单元测试
```

## 持久化会话

每次带任务的旧式调用默认创建一个新的会话并保存消息历史，因此不同任务不会悄悄混在
一起。命令行也可以显式续接会话：

```powershell
# 推荐：显示编号菜单，输入数字选择旧会话或新建会话
coding-agent --workspace C:\path\to\project --select-session

# 继续该工作区最近一次会话
coding-agent --workspace C:\path\to\project --continue "检查刚才的修改"

# 不带任务时进入连续交互，空行、exit、quit 或 EOF 退出
coding-agent --workspace C:\path\to\project --continue

# 明确开始一个全新会话
coding-agent --workspace C:\path\to\project --new-session "实现下一个功能"

# 手动指定新会话标题；未指定时使用第一次任务生成标题
coding-agent --workspace C:\path\to\project --title "分页功能" "为 API 增加分页参数"

```

`--select-session` 会按最近更新时间列出会话标题。输入会话前的编号即可进入连续交互，输入
`0` 新建会话，输入空行、`q`、`quit`、`exit` 或 EOF 取消。也可以在选择命令末尾附带一个
任务，选择后只执行该任务并退出。

会话按规范化工作区路径的 SHA-256 前缀分目录保存。Windows 默认位置为
`%LOCALAPPDATA%\coding-agent\sessions`；其他平台优先使用 `$XDG_DATA_HOME`，否则回退到
`~/.local/share/coding-agent/sessions`。文件通过临时文件加替换原子写入，损坏的 JSON 会
报告为会话错误。历史会限制最大消息数，并在保存前递归脱敏已知 `LLM_API_KEY`，所以 API
Key 不会写入会话文件。每个会话同时保存标题和唯一 ID；自动标题取第一次用户任务、合并
换行与多余空格，并限制为 60 个字符。旧版会话没有标题字段时，会在读取时根据第一条用户
消息生成标题。

这是本机历史，不是模型服务端记忆。继续会话时，受限的消息窗口才会发送给模型；消息中的
文件内容和工具输出仍可能发送到配置的第三方兼容接口，请按服务商的数据政策使用。

常用运行参数：

- `--max-steps 20`：最多请求模型的轮数，达到限制后以非零状态退出；
- `--command-timeout 20`：单条命令最大秒数。模型可以请求更短超时，但不能超过此值。

执行期间，进度写入标准错误，最终回答写入标准输出。成功完成返回退出码 0；配置错误
返回 2；模型错误或达到最大步数返回 1。

## 本地工具

- `list_files`：递归或非递归列出目录，不跟随目录符号链接；
- `read_file`：读取 UTF-8 文本，过长内容按配置截断；
- `write_file`：写入完整 UTF-8 内容，并按需创建父目录；
- `replace_text`：仅当旧文本恰好出现一次时做精确替换；
- `run_command`：使用结构化 `argv` 和 `subprocess.run(..., shell=False)` 执行命令，
  同时记录退出码、标准输出、标准错误和超时状态。

命令输出合并后默认最多返回 20,000 个字符，超出时保留头尾并明确标记截断。

## 安全边界

文件路径必须是工作区相对路径，不能是绝对路径，也不能包含 `..`。每次访问前会解析真实
路径并检查它仍位于工作区内，因此指向外部的符号链接会被拒绝。命令工作目录使用相同的
检查。子进程环境会移除 `LLM_API_KEY`，模型请求错误也会对已知 API Key 做脱敏。

工作区绝对路径不会写入 system prompt，但用户任务、模型消息以及工具返回的文件内容和命令
输出会发送到所配置的第三方模型服务。运行前应确认服务商的数据处理政策，并避免让 Agent
读取不应离开本机的文件。

这些限制不是操作系统级沙箱。命令参数本身可以引用工作区外路径，子进程仍拥有当前用户
的文件和网络权限，并继承除 `LLM_API_KEY` 外的其他环境变量；模型也可能把 shell 程序本身
作为普通可执行文件调用。`write_file` 会覆盖目标文件，目前也没有逐条操作确认或自动回滚。
因此只应在可信任务、可备份或受版本控制的工作区中运行，并在运行前清理其他敏感环境变量。

## 测试与检查

测试不需要网络、真实 API Key 或已安装的 `openai` 包。fake 模型覆盖多轮工具调用、一轮
多个调用、最大步数、未知工具和非法参数；工具测试覆盖路径逃逸、文件读写、命令成功、
非零退出、启动失败与超时；适配器测试覆盖多个 tool calls 和错误脱敏。

```powershell
python -m pytest -q
ruff check src tests
mypy src
python -m compileall -q src tests
```

符号链接逃逸测试在不允许普通用户创建符号链接的 Windows 环境中会跳过；运行时代码仍会
执行真实路径边界检查。

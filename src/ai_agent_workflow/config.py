"""配置解析：从环境变量（可先经 .env 加载）读取模型服务配置。

约定：
- 真实密钥只来自本地 .env 或系统环境变量，绝不写进代码或仓库；
- mock 模式不需要任何真实配置；
- real 模式必须提供 LLM_BASE_URL、LLM_API_KEY、GENERATION_MODEL；
- 日志与配置摘要都不包含明文密钥（见 mask_secret 与 Settings.describe）。
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .errors import ConfigError

logger = logging.getLogger(__name__)

MOCK_MODE = "mock"
REAL_MODE = "real"
MODES = frozenset({MOCK_MODE, REAL_MODE})
RULE_PLANNER = "rule"
LLM_PLANNER = "llm"
PLANNER_MODES = frozenset({RULE_PLANNER, LLM_PLANNER})
DEFAULT_TIMEOUT_SECONDS = 30

#: 项目根目录（src/ai_agent_workflow/config.py -> 上两级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """模型服务配置快照。"""

    mode: str
    base_url: str | None
    api_key: str | None
    generation_model: str | None
    timeout_seconds: int
    # ---- 平台扩展（v0.3）：应用信息、持久化目录与规划器选择 ----
    app_name: str = "AI Agent Execution Platform"
    app_version: str = "0.3.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    api_prefix: str = "/api"
    data_dir: Path = field(default=Path("data"))
    database_url: str | None = None
    planner_mode: str = RULE_PLANNER

    @property
    def ai_configured(self) -> bool:
        """LLM Planner 是否具备调用条件（三项齐全）。"""
        return bool(self.base_url and self.api_key and self.generation_model)

    @property
    def resolved_database_url(self) -> str:
        """默认 SQLite（DATA_DIR/agent.db）；环境变量可指向 PostgreSQL。"""
        if self.database_url:
            return self.database_url
        db_file = (self.data_dir / "agent.db").as_posix()
        return f"sqlite:///{db_file}"

    def ensure_directories(self) -> None:
        """创建运行期必需的目录（幂等）。"""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        """返回可安全打印/记录的配置摘要（不含明文秘密）。"""
        return (
            f"mode={self.mode} generation_model={self.generation_model or '未设置'} "
            f"api_key={mask_secret(self.api_key)} base_url={self.base_url or '未设置'} "
            f"timeout_seconds={self.timeout_seconds} planner={self.planner_mode}"
        )


def mask_secret(value: str | None) -> str:
    """把秘密值转成可安全展示的掩码；空值显示“未设置”。"""
    if not value:
        return "未设置"
    if len(value) <= 8:
        return "***"
    return f"{value[:2]}***{value[-2:]}"


def _optional(raw: str | None) -> str | None:
    """去掉首尾空白；空字符串视为未设置。"""
    if raw is None:
        return None
    text = raw.strip()
    return text or None


def _parse_timeout(raw: str | None) -> int:
    """解析 LLM_TIMEOUT_SECONDS；空值用默认值，非法值抛 ConfigError。"""
    text = _optional(raw)
    if text is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = int(text)
    except ValueError as exc:
        raise ConfigError(f"LLM_TIMEOUT_SECONDS 必须是正整数秒，当前为 {text!r}") from exc
    if value <= 0:
        raise ConfigError(f"LLM_TIMEOUT_SECONDS 必须是正整数秒，当前为 {text!r}")
    return value


def _env_mapping() -> dict[str, str]:
    """生产读取来源：先加载当前目录的 .env（不覆盖已有环境变量），再读 os.environ。"""
    load_dotenv()
    return dict(os.environ)


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """从环境变量构建 Settings；测试可传入 environ，缺省读取真实环境。"""
    env = dict(environ) if environ is not None else _env_mapping()
    mode = (env.get("AGENT_MODE") or MOCK_MODE).strip().lower()
    base_url = _optional(env.get("LLM_BASE_URL"))
    api_key = _optional(env.get("LLM_API_KEY"))
    generation_model = _optional(env.get("GENERATION_MODEL"))
    timeout_seconds = _parse_timeout(env.get("LLM_TIMEOUT_SECONDS"))
    planner_mode = (env.get("PLANNER_MODE") or RULE_PLANNER).strip().lower()
    database_url = _optional(env.get("DATABASE_URL"))
    data_dir = Path(_optional(env.get("DATA_DIR")) or "data")
    if mode not in MODES:
        raise ConfigError(f"AGENT_MODE 必须是 {MOCK_MODE} 或 {REAL_MODE}，当前为 {mode!r}")
    if planner_mode not in PLANNER_MODES:
        raise ConfigError(
            f"PLANNER_MODE 必须是 {RULE_PLANNER} 或 {LLM_PLANNER}，当前为 {planner_mode!r}"
        )
    if mode == REAL_MODE:
        values = {
            "LLM_BASE_URL": base_url,
            "LLM_API_KEY": api_key,
            "GENERATION_MODEL": generation_model,
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ConfigError(
                f"AGENT_MODE={REAL_MODE} 缺少配置项：{', '.join(missing)}"
                "（密钥只写入本地 .env，绝不提交；参考 README 与 .env.example）"
            )
    settings = Settings(
        mode=mode,
        base_url=base_url,
        api_key=api_key,
        generation_model=generation_model,
        timeout_seconds=timeout_seconds,
        data_dir=data_dir,
        database_url=database_url,
        planner_mode=planner_mode,
    )
    logger.info("配置加载完成：%s", settings.describe())
    return settings

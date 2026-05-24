"""src/agent/utils/report_parser.py — 多模态报告解析共享逻辑(DEV_SPEC §4.1.2 ①.5 / ⑨)。

①.5 analyze_initial_reports 与 ⑨ process_exam_result 都需要"file_ref → ReportFinding"
这条管道,封装在这里复用。

工作流:
  file_refs[]
    → 用 report_loader.load_report 按需加载(图片转 base64,PDF 直传)
    → 组装多模态 LangChain HumanMessage(text prompt + image_url 内容块)
    → llm.with_structured_output(ReportFindings, method="json_mode") 调用(中安全等级,§9.1 模板)
    → ReportFindings.findings(每份报告一项)
    → 节点代码补 report_index(对应 exam_reports 下标),写回 State

LLM 模型:走 `settings.llm.VISION_*` 三件套(默认 qwen3.5-plus,DashScope 原生
多模态;参考 `scripts/figure_enrichment_generation.py` 的 figure 增强流水线)。
主链路走 DeepSeek(廉价),不支持视觉,所以本模块必须独立切到 DashScope。
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage

from config.settings import settings
from src.agent.schemas.report_parser import ReportFindings
from src.agent.utils.report_loader import load_report
from src.common.metrics import _attempts, _failures, _latency, retry_observer
from src.models.llm_client import get_llm
from src.prompts.agent import build_report_parsing_prompt


_logger = logging.getLogger(__name__)


def _build_multimodal_message(file_refs: list[str], prompt_text: str) -> HumanMessage:
    """把 prompt_text + 各报告组装成 LangChain HumanMessage(content 列表形态)。"""
    content: list[dict] = [{"type": "text", "text": prompt_text}]
    for ref in file_refs:
        try:
            loaded = load_report(ref)
        except (FileNotFoundError, ValueError) as e:
            _logger.warning("report load failed (%s): %s — skipping", ref, e)
            continue
        if loaded["kind"] == "image":
            content.append(
                {"type": "image_url", "image_url": {"url": loaded["data_uri"]}}
            )
        elif loaded["kind"] == "pdf":
            # OpenAI-compatible providers handle PDF as document part; format varies
            # 不同 provider PDF 块格式不一(OpenAI v2 用 file,DashScope 走 application/pdf
            # data URI),这里给 base64 data URI 兜底,provider 不支持时 LLM 会忽略
            import base64
            b64 = base64.b64encode(loaded["data"]).decode("ascii")
            content.append(
                {
                    "type": "image_url",  # 用 image_url 通道兜底,部分 provider 兼容
                    "image_url": {"url": f"data:application/pdf;base64,{b64}"},
                }
            )
    return HumanMessage(content=content)


def parse_reports(file_refs: list[str], hint: str | None = None) -> list[dict]:
    """主入口:接收文件引用列表,返回结构化 finding 列表(已补 report_index)。

    Args:
        file_refs: 落盘后的文件路径 list
        hint: 可选上下文,如 "这组报告是 '抽血化验(空腹8h)',期望含 血常规、肝功能、淀粉酶";
              ⑨ 按 group 分别调用时传,⑩ 解析时更准。①.5 入站混传不带。

    Returns:
        list[dict],每项形态见 spec §4.1.1 report_findings 字段:
        `{"report_type", "report_date", "report_index", "abnormal_values",
          "impressions", "positive_findings", "negative_findings"}`

    空 file_refs → 直接返回 [],不发起 LLM 调用(spec §4.1.2 ①.5 early return)。
    """
    if not file_refs:
        return []

    node = "analyze_reports"
    schema = "ReportFindings"
    prompt = build_report_parsing_prompt(num_reports=len(file_refs), hint=hint)
    message = _build_multimodal_message(file_refs, prompt)

    _attempts.labels(node=node, schema=schema).inc()
    t0 = time.perf_counter()
    try:
        chain = (
            get_llm(
                model=settings.llm.VISION_MODEL_NAME,
                base_url=settings.llm.VISION_BASE_URL,
                api_key=settings.llm.VISION_API_KEY,
                # 2026-05-24:对齐 ⑩ diagnose 配置(评测验证的 timeout=300/max_tokens=16384)
                # qwen3.5-plus thinking 模式 reasoning+content 共享 budget,2048 默认不够
                timeout_seconds=300,
                max_tokens=16384,
            )
            .with_structured_output(ReportFindings)
            .with_retry(stop_after_attempt=3)
        )
        result: ReportFindings = chain.invoke(
            [message],
            config={
                "callbacks": [retry_observer],
                "metadata": {"node": node, "schema": schema},
            },
        )
    except Exception as e:
        _failures.labels(
            node=node, schema=schema, exception_type=type(e).__name__
        ).inc()
        _logger.error(
            "[%s] report parsing failed (%d files): %s",
            node, len(file_refs), e, exc_info=True,
        )
        # spec §9.3:本调用中安全级,但兜底策略是"该份报告标记解析失败,report_findings
        # 不追加",流水线继续(降级为无该报告证据)。我们对**整批**做兜底:返回空列表
        # 让调用方继续,而不是抛错终止整个 Agent 流程
        return []
    finally:
        elapsed = time.perf_counter() - t0
        _latency.labels(node=node, schema=schema).observe(elapsed)
        _logger.info("[%s] elapsed=%.2fs", node, elapsed)

    # 节点代码补 report_index(对应 exam_reports 下标);LLM 输出 finding 数若与 file_refs
    # 数不一致,以 LLM 输出为准截到合法范围(spec 不强制 1:1)
    out: list[dict] = []
    for i, finding in enumerate(result.findings):
        record = finding.model_dump()
        record["report_index"] = i if i < len(file_refs) else len(file_refs) - 1
        out.append(record)
    return out


def parse_reports_parallel(
    tasks: list[tuple[list[str], str | None]],
) -> list[list[dict]]:
    """对多组报告**并行** VLM 解析。每个 task = (files, hint) 独立调一次 parse_reports。

    VLM 调用是远程 API I/O bound,顺序跑会线性累加(实测 3 组各 40s/60s/40s = 140s)。
    用 ThreadPoolExecutor 并行,总耗时 ≈ max(各 task elapsed),典型场景省 50%+。
    每 task 独立 LLM 调用,无共享状态,线程安全。

    Returns:
        list[list[dict]],与 tasks 同长度同顺序;每项是该 task 的 findings list。
        某 task 失败 → 对应位置返空 list(沿用 parse_reports 单调用的 fail-soft 行为)。
    """
    if not tasks:
        return []
    if len(tasks) == 1:
        # 单组无需起线程池,直接顺序调,省一层开销
        return [parse_reports(*tasks[0])]

    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futures = [ex.submit(parse_reports, files, hint) for files, hint in tasks]
        # 按提交顺序收结果(.result() 阻塞等单个完成,但线程池里其他 task 已并发开跑)
        return [f.result() for f in futures]

# 3. RAG系统设计：
## 3.1 数据摄取

### 3.1.1 数据加载及处理

#### 文档解析器选型：MinerU

医疗场景的文档具有高度复杂性与专业性，对解析精度要求极高。选用 MinerU 作为文档解析器，核心优势如下：

| 能力 | 说明 |
|------|------|
| 扫描件与影像报告支持 | 医院文档大量以扫描 PDF 形式存在（如检验报告、病历归档），MinerU 内置高精度 OCR 引擎 |
| 复杂表格高精度还原 | 基于深度学习的表格识别模型，可准确还原行列结构，确保表格数据进入向量库后语义完整 |
| 医学公式与专业符号识别 | 支持 LaTeX 格式的公式输出，准确提取计量单位、化学式及统计公式 |
| 图文混排文档处理 | 具备多模态解析能力，可对图表进行结构化处理，而非直接丢弃 |

#### MinerU 输出结构

通过命令行运行 MinerU 解析后，输出目录结构如下(子目录名随 backend 而变,hybrid-auto-engine → `hybrid_auto/`,vlm-auto-engine → `vlm_auto/`,pipeline → `pipeline_auto/`):

```
/project_folder/mineru_output/target_document/{backend}_auto/
├── images/                                   # 提取的图片资源(SHA 命名)
├── target_document.md                        # 最终 Markdown 输出(给人预览,不直接灌库)
├── target_document_content_list_v2.json      # ⭐ 下游 chunking 实际消费(页级嵌套,块带完整语义)
├── target_document_content_list.json         # v1 扁平结构(向后兼容,本项目不消费)
├── target_document_middle.json               # 中间解析结果(含 spans/score/lines,体积大,默认仅留磁盘不入库)
├── target_document_model.json                # 模型原始推理(归一化 bbox,默认仅留磁盘不入库)
├── target_document_origin.pdf                # 原始 PDF 拷贝
└── target_document_layout.pdf                # 版面分析框图(肉眼检查用)
```

#### 多类型内容处理策略

项目数据源含大量多类型内容(表格、流程图、统计图、影像图、化学结构、公式)。**2026-05-03 拍板**:只保留 **table / figure** 两类做 chunk(figure 涵盖 mineru type=image 中 sub_type=flowchart 的子集 + type=chart 全部;后者识别质量差,统一按图片处理),其他全部在 chunking 阶段内存过滤丢弃(详细决策依据见 `非项目本体/图表等处理方式.md`)。

**单行多列设计(2026-05-12 修订自原"源 chunk + summary chunk"双行架构)**

每张图/表 mineru block 对应 chunks 表的**一行**,该行同时承载原始 payload 与 LLM 生成的医学陈述:

- `chunk_raw_text`:caption + html + footnote(table) / caption + footnote(figure;**不写入 mineru mermaid / markdown**,见下"为什么 figure 不存 mermaid/markdown"),作为 **BM25 sparse 索引来源**;table 的 html 同时也是命中后图表展示 payload
- `medical_statement`:LLM 看截图/html 后生成 100-300 字医学陈述(把图表数据线性化为陈述句),作为 **dense `original` 向量来源**(因 figure 的 chunk_raw_text 只有 caption 信息量太低、table 的 html 作 dense 表达力不足)
- `summary` / `hypothetical_questions`:走标准 §3.1.3 enrichment(同 child),分别驱动 dense `summary` / `question` 向量
- `image_path`:截图相对路径,命中后供消费 LLM(node ⑩ diagnose 等)直接喂图(figure 类)

| 类型 | chunk_raw_text 含 | medical_statement 生成 LLM |
|---|---|---|
| `table` | caption + html + footnote | **text LLM** 看 html → 100-300 字医学陈述 |
| `figure` | caption + footnote(**不含** mermaid / markdown) | **vision LLM** 看截图(原始 mineru mermaid / markdown 可选作辅助 prompt 提示,但**不入库**)→ 100-300 字医学陈述 |

**为什么 figure 不存 mermaid / markdown 进 chunk_raw_text**:mineru 把 flowchart 转 mermaid 与 chart 转 markdown 数据表的转录质量都很差(~14% mermaid 链式爆炸、心电图 markdown 几乎完全失效,见下文"LLM 选型"),从 ingest 到检索我们从不消费这两路文本,只让 vision LLM 看截图。把垃圾 mermaid/markdown 灌进 chunk_raw_text 会污染 BM25 倒排索引(把噪音 token 当关键词命中)且占用存储,因此明确**不写入 PG**。原始转录由 mineru 输出文件 / raw_documents 保留作审计,chunks 表只存"可用"的文本。

**消费侧 LLM 选型对齐 ingest 侧**:命中 figure 时给消费 LLM 直接喂截图(payload 里的 `image_path`);命中 table 时只喂 html(text LLM 已足够)。

**为什么改单行**(2026-05-12 重构):
- 原"源 + summary"双行设计造成 chunks 表内 figure/table 占用 2 倍行数,`heading_path` / `source_id` / `parent_chunk_id` 等元数据重复存储
- summary chunk 命中后需走 `linked_chunk_id` 回查源 chunk 取 payload,多一次 JOIN
- summary 与源本质是同一图表的两种表示(关键词索引 vs 语义索引),不应建模成两个独立 chunk
- 单行多列后:chunk_id 唯一指向一张图/表,所有字段同行直读;BM25 sparse 由 chunk_raw_text 列驱动,dense 向量由 medical_statement / summary / hypothetical_questions 列驱动,Milvus 多向量结构不变(见 §3.1.5)

**丢弃五类的理由**:
- `image - chemical / natural_image / None`:CT/MRI/化学截图喂多模态 LLM 也用不好(教科书 200KB 截图非诊断级 DICOM,缺乏 SMILES/InChI 等结构化标准)
- `image - text_image`:实测 90% 是出版社推广页(扫码激活/在线服务),少量解剖标注图无法可靠从噪音中分离
- `equation_interline`:量太小(典型一本书 8-50 个),公式通常已在所属段落文字描述里带过,单独入库 ROI 极低

**为什么不扩展到被丢类型中的"医学示意图"**(2026-05-08 实测后再确认):全 12 本扫被丢的 7475 条 image,caption 关键词命中"分诊潜在高价值"约 ~917 条(解剖位置 505 / 体表病变 228 / 体征查体 136 / 皮肤外观 48,主要被 mineru 错判到 `text_image` / `natural_image` / `None`)。**坚持不扩**,理由三层叠加:
1. **vision LLM 能力分级**:保留的两类(table/figure)是**模式化视觉**(html 表、数据图、流程图都有标准视觉范式),vision LLM 可靠性可控;扩进来的 natural_image / text_image 是**自由形态视觉**(CT/MRI 影像、皮疹照片、解剖摄影),vision LLM 看自由形态医学图像可靠性低一个量级,易引入幻觉污染
2. **正文已覆盖**:解剖位置 / 体表分区 / 皮疹外观等医学事实在所属节正文里都有等价文字描述(如"胸骨柄位于...第 2 肋对应胸骨角");medical_statement 跟正文 cosine 去重(§3.1.3 阈值 0.85)会大量丢这些条目,召回增益小但已付出 vision 幻觉风险
3. **前端 UI 与 RAG 召回是两件事**:真正"非看图不可"的类型(典型皮肤病例图)价值在前端给患者视觉对比,这是产品 UI 功能,不归 RAG chunk 库承担

**关键约束(为什么必须独立成 chunk,不能 inline 进父子文本块)**:
1. mineru block 在 `content_list_v2.json` 里按版面位置排,**不严格按引用关系**(实测一页里 table 在页头,讨论它的段落在页中段)
2. 实测 18% table、7% flowchart 单 block 字符数 > 1200 字父块目标(max 7080),**强行 inline 会撑爆父块大小**
3. 因此通过 `heading_path_id` 做关联召回(图表与同节正文 chunk 共享 heading_path_id,不 inline 但同节可批量回查,见 §3.2 规则 3),而不是 inline 合并

**LLM 选型(2026-05-08 实测后修订)**:**按 mineru 转录可靠性分流**——

- **table → text LLM 看 html**:mineru 表格 OCR 质量稳定(html 行列完整、caption/footnote 单独成字段),text LLM 看 html 是 text→text 任务,成本低质量稳
- **figure(原 chart 子集)→ vision LLM 看截图**:实测 mineru 把曲线图 OCR 成 markdown 时**信息丢失严重**(心电图 36 条 markdown 仅标各导联峰值高低,完全不含 P 波/QRS/ST 段细节;部分 forest plot / heatmap 视觉对比信息也丢)。markdown 同时作为辅助 prompt 提示给 vision LLM(让它有数据点参考,但以截图为真值)
- **figure(原 flowchart 子集)→ vision LLM 看截图**:实测 mineru 转 mermaid 在复杂分子机制图 / 蛋白结构图被错判为 flowchart 时会**链式爆炸**(单 label 重复 50+ 次)等系统性缺陷,~14% mermaid 不可用;此外 mermaid 看不出"逻辑顺序错"等隐蔽错误。mermaid 同样作为辅助 prompt 提示

成本(全 12 本一次性 ingest):table 2980 × $0.005 + figure 1046(原 chart 320 + 原 flowchart 726)× $0.05 ≈ **$67**(¥480)。比"统一 text LLM"贵 3 倍但消除 mineru 转录质量风险,生产消费侧也对齐这套(命中 figure 直接喂截图)。

---

> **关于图像内容理解的设计原则(2026-05-08 修订)**:
> 1. **教科书图配文约定**:图表达的核心信息一般在周围正文里重复,正文召回往往可以兜底,但**对纯数据图(检验值表、诊断流程图)正文无法替代** —— 这是引入 `medical_statement` 列承载语义化陈述的根本理由
> 2. **按 mineru 转录可靠性分流 LLM**:table 用 text LLM 看 html(mineru OCR 稳)、figure 用 vision LLM 看截图(mineru markdown/mermaid 转录有系统性缺陷,实测 ~14% mermaid 链式爆炸,心电图 markdown 完全失效)。原"统一 text LLM"的设计已废弃 —— 实证表明垃圾结构化文本喂 text LLM 比让 vision LLM 看真截图更易引入幻觉
> 3. **medical compliance**:诊断依据可追溯到权威文本(`chunk_raw_text` 仍存 mineru 结构化转录,`image_path` 指向截图,可校验);`medical_statement` 是召回辅助层,不作为权威依据出现在最终诊断引用中
>
> 患者实时上传的检查报告走独立路径(见 Agent ①.5 / ⑨),Vision LLM 直读并结构化为 `report_findings`,与本节知识库摄取不共用机制。

#### MinerU 解析的已知限制与下游补救

实测 hybrid-auto-engine 在医学教科书上整体质量优秀(表格 HTML 行列完整、caption/footnote 单独成字段、chart 把曲线图 OCR 成 markdown 数据表、page_header/footer/number 单独识别可干净过滤),但有两处**系统性限制**需要在加载/切分阶段主动处理。这两点不是某本书的特例,换 backend 或重跑无法解决,**必须在代码层补救**。

**限制 1:image 块的 `content` 字段质量随 sub_type 差异巨大(loader + chunking 两阶段处理)**

MinerU 对 `type=image` 不同 sub_type 的 `content` 字段质量差别很大,实测:

| sub_type | content 字段质量 | 处理 |
|---|---|---|
| `flowchart` | **mermaid graph 代码**,节点+连线还原,质量高 | **保留**(在 chunking 阶段进入 `chunk_type='figure'` 单行,vision LLM 生成 `medical_statement`) |
| `text_image` | OCR 文本,但 90% 是出版社推广页噪音(如"扫码激活/在线服务") | 丢弃 image content + 整块在 chunking 过滤 |
| `chemical` | 自然语言描述非 SMILES,质量参差 | 丢弃 image content + 整块在 chunking 过滤 |
| `natural_image / None` | VLM 泛泛描述("Abstract red background"等),无医学价值 | 丢弃 image content + 整块在 chunking 过滤 |

**Loader 阶段(mineru_loader.py)**:为兼容历史方案 + 保 raw_documents 完整,**当前继续对所有 `type=image` 块统一删除 `content.content` 字段**(包括 flowchart 的 mermaid)。flowchart 的 mermaid 在下游 chunking 阶段重新从 raw 取(因为 raw_documents 保留的是删 image VLM 文本前的原始 content_list,但本字段已删)。

**修正方案(待实现)**:loader 改为按 sub_type 选择性删除,只删 `text_image / chemical / natural_image / None` 的 content,**保留 flowchart 的 mermaid**。这一改动可以同步进行也可以延后(chunking 阶段从 mineru 输出文件直接读 mermaid 也行,raw_documents 内容可不严格同步)。

注意:`type=table` 和 `type=chart` 块的 `content` 字段必须**保留**(分别是 HTML 表格和 markdown 数据表,质量高且为信息核心载体)。

**Chunking 阶段(chunking.py)**:按上文"多类型内容处理策略"做内存过滤 — table / chart / image-flowchart 三种 mineru block 各产生 chunks 表**一行**(table 入 `chunk_type='table'`,后两者归并入 `chunk_type='figure'`);image 其他 sub_type + equation_interline 直接 continue 不进 chunks 表。mineru 产物文件永久不变,丢弃只发生在内存过滤,决策可逆。

**限制 2:`title.level` 全部退化为 1(已弃案:正则重建 → 改用目录权威清单)**

MinerU 输出的 `content_list_v2.json` 中所有 title 块的 `level` 字段统一是 1,丢失原始多级标题结构。同时实测还发现:**同一格式的 anchor 在同一书内被 mineru 标 type=title 还是 type=paragraph 完全不一致**(POC §1.3 bug 5),导致即使"按文本正则重建 level"也补救不全。

**已弃案**:早期方案是按文本正则匹配重建 level(篇/章/节/数字编号 → L1/L2/L3/L4),但 POC 验证发现:
- 不同书的章节命名约定差异极大(《临床用药指南》99.9% fallback 因每药品名独成 title),正则归级跨书不可复用
- 即使本书命中率 80%,production code 维护代价过高
- 节内子节(【】/(一)/1.) mineru type 标记完全不可信,正则重建解决不了节内切分问题

**新方案**(2026-05-03 用户拍板):**完全放弃 mineru `title.level`,改用"目录权威清单"思路** —— 从 mineru 目录页(`page_header` 含"目录")抽出本书完整目录结构作为唯一层级真值,正文匹配时 fuzzy match 到目录条目得到权威 level/heading_path。详见 §3.1.2 切分主流程。

**给消费者的结论**:任何 chunking / 父子索引代码都不应该读 `title.level` 字段(永远是 1,无意义)。父块层级与边界由"目录字典 + 正文匹配"决定。




## 3.1.2 chunking(目录权威清单 + 三遍切 + size 驱动子块)

**核心方法论**(2026-05-03 经《内分泌代谢病学第4版上册》POC 验证):

- **完全不依赖 mineru `title.level`**(永远是 1,无意义,见 §3.1.1 限制 2)
- **不用 `RecursiveCharacterTextSplitter`**(已弃案)
- 父块由"书的目录结构"切(节 + 节内三遍切【】+(一)+1.),子块由"父块大小"切(size 累积驱动)
- 父子结构**仅对真正大的父块有意义**:小父块直接当 child(避免 degenerate "父=子")
- POC 实现与详细方法论见 [`scripts/poc_chunking_内分泌代谢病学_第4版上册/METHODOLOGY.md`](scripts/poc_chunking_内分泌代谢病学_第4版上册/METHODOLOGY.md);production 实现位 `src/rag/ingestion/chunking.py`(待 port)

### Block 白名单与文本抽取规则(消费 `content_list_v2` 前必读)

`raw_documents.content_list` 的真实结构与 10 种 `block.type` 见 §2.4.4.1。chunking 阶段**必须**按下表分类处理。

**进入文本父子分块的 3 种 type(累积进 parent/child 文本)**:

| type | 抽取规则(从 `block.content` 取正文) | 用途 |
|---|---|---|
| `title` | 拼 `title_content[].content`(`type=text` 子项)。**忽略 `level` 字段**(永远 1,见 §3.1.1 限制 2),改用目录权威清单匹配确定 level | 父块边界候选(节级匹配)+ 节内子标题边界候选(【】/(一)/1. 正则) |
| `paragraph` | 拼 `paragraph_content[].content`(`type=text` 子项) | 父块/子块的正文输入 |
| `list` | 递归遍历 `list_items[].item_content[].content`(深 4 层),按 `list_type` 决定是否加 `1./- ` 前缀。**整体作为不可分语义单元**,首项 `1.` 不当子标题切点 | 父块/子块累积输入 |

**进入图表 chunk 路径的 3 种 type(每个 block 生成 chunks 表一行,单行多列设计)**:

| type / sub_type | chunks 表 `chunk_type` | 入库字段(同一行) |
|---|---|---|
| `table` | `table` | `chunk_raw_text` = caption + html + footnote;`image_path` = 截图;`medical_statement` = text LLM 看 html 生成 100-300 字医学陈述 |
| `chart` | `figure` | `chunk_raw_text` = caption + footnote(markdown 数据表**不写入**);`image_path` = 截图;`medical_statement` = vision LLM 看截图生成医学陈述 |
| `image` + `sub_type=flowchart` | `figure` | `chunk_raw_text` = caption + footnote(mermaid **不写入**);`image_path` = 截图;`sub_type` = 'flowchart';`medical_statement` = vision LLM 看截图生成医学陈述 |

图表行走标准 §3.1.3 enrichment(再生成 `summary` + 3 条 `hypothetical_questions`),`embedding_status='pending' → done`,**共享同一 `heading_path_id` 与同节文本 chunk** 关联召回。理由与详细架构见 §3.1.1 "多类型内容处理策略"。

**全部丢弃的 type / sub_type(直接 continue,不进 chunks 表)**:

| type / sub_type | 丢弃理由 |
|---|---|
| `page_header` / `page_footer` / `page_number` | 页眉/脚/页码,与正文无关 |
| `image` + `sub_type ∈ {chemical, text_image, natural_image, None}` | mineru 转录质量低且喂多模态 LLM 也用不好,详见 §3.1.1 |
| `equation_interline` | 数量小(典型一本书 8-50 个),公式通常已在所属段落文字描述里带过 |

**实现位置**:`src/rag/ingestion/chunking.py` 的 block 主循环按上表三类路由(进入文本父子 / 进入图表独立 / 丢弃)。

**关键提醒**:文本父子分块的 chunk(parent / child)不会 inline 包含 table / chart / mermaid 内容(这些走独立 chunk 路径)。在召回链路上,通过 `heading_path_id` 命中节内文本 chunk → 同节内 figure/table chunk 在同一 `heading_path_id` 下,可批量回查并把 `chunk_raw_text`(table=html,figure=caption)+ `medical_statement` + `image_path` 合并入 LLM context(见 §3.2 规则 3);figure 类还按 `image_path` 直接喂截图给消费 LLM。

### 切分主流程(4 步)

#### Step 1:目录权威清单提取

扫 `content_list_v2` 的目录页(`page_header` 含"目录"的页),对所有 paragraph/title/list block 抽行,按本书的 anchor pattern 分级(L1 篇 / L2 章 / L3 节 / L4 数字编号):

- 跨条目粘连拆分(mineru 会把"第2节...56第3节..."焊一行)
- 黑名单剔除("上册/下册/全书概览/目录")
- normalize:删 `\n` 残留,折叠空白,剥页码尾,合并节号
- strict_key:进一步去掉所有空白(应对 mineru 中文/ASCII 间空格风格不一致)

输出:`{normalized_title: (level, parent_path)}` 字典,作为后续匹配的权威真值。

**注意**:不同书的 anchor pattern 差异大(《用药指南》是药典结构,99.9% 命中失败),每本书需单独适配 pattern。

#### Step 2:正文节边界匹配(REAL_START 选取)

正文范围 `page_idx > max(toc_pages)`,对每个 type=title block 做 strict_key 匹配字典,加 3 类预处理:

- **A1 章合并**:mineru 把"第N章"和"章名"拆成两个 title block,合并
- **A2 篇前缀重建**:篇标题丢失"第N篇"前缀,从字典 L1 反查 alias 补回
- **A3 mini-TOC paragraph**:扩展资源等被标 paragraph,严格双条件采纳(末尾带页码 + 命中字典)

每个字典 title 在正文里可能出现多次(章/篇页 mini-TOC + 真章节起始),按以下规则选 REAL_START:
1. 优先级 1:按文档顺序最后一个满足"强信号"的 match(`PART_REBUILT/CHAP_MERGED` 或 `AS_IS gap_chars≥50`)
2. 优先级 2:都没强信号 → 取最后一次出现位置

输出:159 个节起点位置(节级原父块边界)。

#### Step 3:全书层面截断 + 节内参考文献丢弃

**书末截断**:扫 flat block 序列,第一个命中 `BODY_END_MARKERS = ('中文名词索引', '英文缩略语索引', '彩色插图')` 的 title 即截断,后续全丢。

**参考文献丢弃**:节内扫到 `^参考文献\s*$` 标题即截断,该位置及之后的所有 block 全部丢弃(包括 ref 条目 + 紧随其后的扩展资源占位列表)。理由:英文学术 ref 与中文医学查询语义不匹配,扩展资源是外部链接占位,均无 RAG 召回价值。

#### Step 4:父块构建(节内三遍切 + 严格层级合并)

每个节本身就是默认父块。如果节字符 > **`PARENT_SPLIT_THRESHOLD = 4000`**,启动三遍逐级细化:

| Pass | 触发 | 加边界 pattern | level |
|---|---|---|---|
| 1 | 段 > 4000 字 | `^【.+?】` | 1 (BRACE) |
| 2 | Pass 1 后段仍 > 4000 字 | `^[（(][一二三四五六七八九十百]+[)）]` | 2 (PAREN) |
| 3 | Pass 2 后段仍 > 4999 字 | `^\d+\s*[.、]\s` | 3 (NUM) |

排除:`type=list` block(整体语义单元)、`^表/图\s*[\d-]+`、长度 < 4 字符的残片。

**小父块合并**(< **`PARENT_MERGE_TINY_THRESHOLD = 500`** 字):

严格按层级关系。**吸收方 level ≤ 被吸收方 level**:
- Forward: `cur_level ≤ next_level`(允许同级兄弟、上级吸子主题、节首段;禁止下级跨上级)
- Backward: `prev_level ≤ cur_level`

例:1./2. 不能跨 (一)、(一) 不能跨【】合并;但同节下【】兄弟可以合并(同节都算相关主题)。

#### Step 5:子块构建(size 驱动)

每个父块独立判断:

- **父块 ≤ `CHILD_SPLIT_THRESHOLD = 1200` 字**:**不切**,1 child = parent 整段(避免 degenerate)
- **父块 > 1200 字**:按 mineru block 累积切多 child,目标 `CHILD_TARGET_SIZE = 600` 字
  - 算法:每加一个 block 看"加 vs 不加"哪个 acc_len 更接近 target,选更近的
  - 强制最小 `CHILD_MIN_SIZE = 200` 字:当前累积 < 200 时无视距离判断,force-add 防止孤儿
  - 末段 < target/2 时 backward 并入上一 child
  - 单 mineru block 即使 > target 也独立成 child(block 是不可分的最小语义单元)

子块切分**完全不用标题 pattern**,只看大小 + mineru block 边界。这样父块切法和子块切法解耦,避免 degenerate。

### 关键阈值(目标 ~3000 token 父块,~600 字子块)

实测 Qwen tokenizer 1 token ≈ 1.39 字符。

| 常量 | 值 | 用途 |
|---|---|---|
| `PARENT_SPLIT_THRESHOLD` | 4000 字(~2877 token) | 父块切【】+(一)阈值 |
| `PARENT_PASS3_THRESHOLD` | 5000 字(~3597 token) | 父块切 1./2. 阈值(稍宽,避免过切) |
| `PARENT_MERGE_TINY_THRESHOLD` | 500 字 | 小父块合并阈值 |
| `CHILD_SPLIT_THRESHOLD` | 1200 字(~864 token) | 父块 ≤ 此值不切子块 |
| `CHILD_TARGET_SIZE` | 600 字(~432 token) | 大父块切子块的目标 size |
| `CHILD_MIN_SIZE` | 200 字 | 子块强制最小,< 此值 force-add 防孤儿 |

### 父子索引(Small-to-Big 检索模式)

**设计动机**:医疗知识天然分层(疾病 → 亚型 → 治疗方案 → 剂量/禁忌)。小块精确命中"剂量"时,禁忌证可能在同节另一小块,导致危险遗漏。父块作为完整上下文的兜底锚点,确保临床信息的完整性。

**实现策略**:
1. **父块**:按上述 Step 4 切分,写入 `chunks` 表,`parent_chunk_id = NULL`,**不做向量化**(`embedding_status='skip'`,仅作内容存储)
2. **子块**:按 Step 5 size 驱动切,记录 `parent_chunk_id` 指向所属父块。子块正常进行多向量 Embedding(见 3.1.5)
3. **顶层兜底**:文档开头无标题的散文片段(罕见),`parent_chunk_id = NULL`,精排后直接使用该小块原文
4. **父块 ID 生成**:父块使用与子块相同的 `heading_path_id` 方案 + 后缀("parent")生成稳定 `chunk_id`,幂等逻辑见 §3.1.4
5. **Milvus 不变**:父块不写入 Milvus,向量检索仅针对子块
6. **不变量**:total_parent_len == total_child_len(父子内容完整守恒,任何切分逻辑改动后必须验证 mismatch=0)

### 数据快照(本书 POC 结果,2026-05-03)

| 指标 | 值 |
|---|---|
| 节数(节级原父块) | 159 |
| 父块数(三遍切+合并后) | 1204 |
| 父块 size median / max | 1346 / 5218 字 |
| 子块数 | 3012 |
| 子块 size median / max | 616 / 1528 字 |
| 书末截断丢弃 | 1676 blocks / 20721 字 |
| 参考文献丢弃 | 607 blocks / 16257 字 |
| 父子覆盖完整性 | mismatch=0 (1932461 字) |


## 3.1.3 Transform & Enrichment（结构转换与深度增强）

### 3.1.3.1 结构转换

§3.1.2 chunking 主流程的输出是两类 dict 列表:
- `parents: list[ParentChunk]` — 父块,每条含 `parent_idx / section_title / level / title / head / pg_start / len`
- `children: list[ChildChunk]` — 子块,每条含 `parent_idx / section_title / head / pg_start / len / blocks`

本步骤将 chunk 文本与各阶段元数据整合,写入 `chunks` 表(字段定义详见 §2.4 → chunks 表)。父块 `embedding_status='skip'`,子块进入下游 enrichment / embedding pipeline。

### 3.1.3.2 增强策略

**语义元数据注入 (Semantic Metadata Enrichment)**：

策略：在基础元数据之上，利用 LLM 提取高维语义特征。
产出：为每个 Chunk 通过**单次 LLM 调用**统一生成以下字段，注入到 Metadata 中：
- **Title**（精准小标题）
- **Summary**（内容摘要）：同时作为摘要向量的文本来源（见 3.1.5）
- **Hypothetical Questions**（假设性问题）：针对本 Chunk 内容生成 **3 条**用户可能提出的问题，**期望覆盖三种风格**：① 专业术语+疑问；② 症状口语+推断；③ 场景化人物代入。该字段为 chunk 提供多个语义入口，弥合 query/chunk 的表述差异，提升患者口语 query 的召回率。各字段实际命中的 query 模式与多向量联合兜底链路详见 §3.1.5.1。

> **Tags 字段已删除**(2026-05 决策):spec 原钦定 Tags 用于 §3.2.3 Pre-filter,但 Pre-filter 实际由 LLM 生成的 `tags`(chunk 端)与 LLM 生成的 query 端 tags 做交集过滤,LLM 命名漂移(如 "LDL-C" vs "低密度脂蛋白")让对齐失败概率大,实施后效用低于 source_id pre-filter。已从 `ChunkEnrichmentOutput`、`chunks` 表、Milvus payload schema 全面移除。

**工程特性**：Transform 步骤为原子化操作，每个 Chunk 独立处理，失败时仅需重试该 Chunk，不影响其他已完成的 Chunk。

**chunk_type 区分的 enrichment 路径**:

| chunk_type | enrichment 路径 |
|---|---|
| `parent`(父块) | 不参与本步骤(`embedding_status='skip'`,见 §3.1.2 父子索引) |
| `child`(子块) | 走完整 enrichment(title + summary + hypothetical_questions)。dense `original` 向量来源 = `chunk_raw_text` |
| `table` / `figure` | **单行多列,单次 LLM 调用产出 4 字段**:① `medical_statement`(看源数据生成 100-300 字医学陈述,作为 dense `original` 向量来源,替代 chunk_raw_text);② `title`;③ `summary`(100-150 字方法论摘要);④ `hypothetical_questions`(3 条)。LLM 选型按 chunk_type 分流:`table` 用 text LLM 看 html,`figure` 用 vision LLM 看截图(mermaid/markdown 作辅助提示),详见 §3.1.2 LLM 选型 |

**`medical_statement` 生成阶段的 prompt 约束**(`table` / `figure` 行专用):
- 用医学教科书段落口吻陈述源数据中的医学事实,不要描述源对象本身("图中可见"、"本表列出" 等图说/表说语)
- 长度 100-300 字(超大表可放宽到 500 字)
- 生成后跟所属节正文段落算 cosine 去重:> 0.85 丢弃整行(信息已被正文覆盖,避免冗余 chunk)


## 3.1.4 幂等性设计(Idempotency)

**核心机制**：

三层存储均通过 Upsert 保证幂等写入，同一文档无论被处理多少次，均不产生重复数据：

| 存储层 | Upsert 主键 | 说明 |
|--------|------------|------|
| PostgreSQL `sources` 表 | `source_id` | 同一文档重复导入时直接覆盖，不新增记录（详见 3.1.4.1） |
| PostgreSQL `chunks` 表 | `chunk_id` | 配合 `content_hash` 实现增量更新，内容未变则跳过 Embedding（详见 3.1.4.2、3.1.4.3） |
| Milvus 向量记录 | 派生 ID | 由 `chunk_id` 确定性派生，如 `{chunk_id}_summary`（详见 3.1.6） |

**原子性保证**：Upsert 以 Batch 为单位进行事务性写入。若批次内某条写入失败，整批回滚，不产生部分写入的脏数据，下次重试时整批重新处理即可。


### 3.1.4.1 source_id

`source_id` 是来源文档的唯一标识符，代表文档的 **逻辑身份** 而非特定版本。同一份指南/教材重新上传（内容可能有修订）应命中同一个 `source_id`，触发 Upsert 更新而非新增记录。

**生成规则**：基于文件名的确定性哈希。

**构建步骤**（顺序执行）：

1. **去除扩展名**：取文件名的 stem 部分（如 `2024心力衰竭指南.pdf` → `2024心力衰竭指南`）
2. **normalize**：复用 3.1.4.2 中定义的 `normalize` 函数（Unicode NFC 规范化 → 全角转半角 → 转小写 → 去除首尾空白 → 合并内部连续空白）
3. **SHA-256 哈希并截断**：对 normalize 后的字符串取 UTF-8 编码的 SHA-256，截取前 16 位十六进制字符（64 bit）

```python
import hashlib
from pathlib import Path

def generate_source_id(file_name: str) -> str:
    stem = Path(file_name).stem                          # 去扩展名
    norm = normalize(stem)                               # 复用 3.1.4.2 的 normalize
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
```

**示例**：

| 输入文件名 | normalize 后 | source_id |
|-----------|-------------|-----------|
| `2024心力衰竭指南.pdf` | `2024心力衰竭指南` | `a3b1c9...`（16 位 hex） |
| `2024心力衰竭指南.md` | `2024心力衰竭指南` | 同上（扩展名不影响） |
| `　２０２４心力衰竭指南.pdf`（含全角） | `2024心力衰竭指南` | 同上（全角归一化） |

**设计决策与边界情况**：

- **为什么不用文件内容哈希**：内容一改 `source_id` 就变，无法 Upsert 更新旧记录，导致旧 chunks 变僵尸、新记录重复插入，与幂等设计矛盾。
- **为什么不用随机 UUID**：不可复现，同一文件重复上传会产生两条 source 记录，破坏幂等性。
- **同名不同文档**：会被视为"同一文档的更新"——这是 Upsert 语义下的期望行为。若确实是不同文档，管理员应修改文件名以区分。
- **文件改名**：同一份指南改了文件名会产生新 `source_id`，旧记录不会自动清理。可通过管理后台的知识库变更记录（见 5.2.3.2 `kb_change_log`）提供"合并/替换 source"的操作。
- **碰撞概率**：16 位 hex = 64 bit，在医疗文档量级（千～万级）下碰撞概率可忽略（生日攻击阈值约 2^32 ≈ 40 亿）。

**幂等写入**：每次文档摄取时，以 `source_id` 为主键对 `sources` 表执行 Upsert，更新 `updated_at` 等可变字段，不重复插入记录。


### 3.1.4.2 heading_path_id 的构建


**设计动机**：避免使用绝对位置编码——若使用绝对位置，文档中任意一处修改都会导致其后所有 Chunk 的位置编码全部失效。改用标题路径作为定位锚点，则只有标题本身变更才会影响对应的 `chunk_id`。

**构建步骤**：

**`normalize` 函数定义**

对标题文本执行以下操作（顺序执行）：

1. **Unicode 规范化**：转换为 NFC 形式，统一字符的组合方式
2. **全角转半角**：将全角字母、数字、空格转为对应半角字符（如 `Ａ→A`、`１→1`、`　→ `）
3. **大小写统一**：所有拉丁字母转为小写
4. **去除首尾空白**：trim 前后的空格、制表符
5. **合并内部空白**：将连续的空白字符（空格、制表符）压缩为单个空格

```python
import unicodedata
import re

def normalize(title: str) -> str:
    # 1. Unicode NFC 规范化
    s = unicodedata.normalize("NFC", title)
    # 2. 全角转半角
    s = s.translate(str.maketrans(
        "　！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？"
        "＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿"
        "｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～",
        " !\"#$%&'()*+,-./0123456789:;<=>?"
        "@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_"
        "`abcdefghijklmnopqrstuvwxyz{|}~"
    ))
    # 3. 转小写
    s = s.lower()
    # 4. 去除首尾空白
    s = s.strip()
    # 5. 合并内部连续空白
    s = re.sub(r'\s+', ' ', s)
    return s
```

**设计说明**：
- 中文字符不做额外处理，NFC 已保证其规范性
- 不去除标点符号——标题中的标点（如冒号、括号）可能是有意义的区分因素
- 不做 stemming 或同义词处理，保持哈希的确定性和可复现性

---

**步骤 1：标准化各级标题，生成层级哈希**

将每个层级标题映射成一个稳定标识符（对标题文本规范化后取哈希）：

```
H1_id = hash(normalize(title_level1))
H2_id = hash(normalize(title_level2))
H3_id = hash(normalize(title_level3))
...
更深层的标题以此类推
```

结果为一个层级哈希序列，如 `[H1_id, H2_id, H3_id]`。

**步骤 2：拼接层级哈希，生成 heading_path_id**

将层级哈希按顺序拼接（冒号分隔），再整体哈希一次，得到固定长度的十六进制字符串。**只拼接实际存在的层级**，不补空位：

```
# 两级标题
heading_path_id = SHA256( H1_id + ":" + H2_id )

# 三级标题
heading_path_id = SHA256( H1_id + ":" + H2_id + ":" + H3_id )

# 通用形式
heading_path_id = SHA256( join(":", [H1_id, H2_id, ..., Hn_id]) )
```

**步骤 3：结合相对块索引，生成 chunk_id**

`relative_chunk_index` 为同一标题路径下的 Chunk 顺序编号（从 0 开始），确保同标题下多个 Chunk 各有唯一 ID，代入最终公式即得 `chunk_id`。

**最终公式**：

```
chunk_id = SHA256( source_id + ":" + heading_path_id + ":" + relative_chunk_index )
```

> **父块约定**：父块代表整个 heading 节，不属于任何子块序列，固定使用字符串 `"parent"` 作为 `relative_chunk_index` 参与哈希：
> ```
> parent_chunk_id = SHA256( source_id + ":" + heading_path_id + ":" + "parent" )
> ```
> 这确保同一 heading 节下只会产生一个父块 ID，且与所有子块 ID 不冲突。

> **图表 chunk 约定**(C5 enrichment 落 PG 阶段使用):mineru 抽出的 `figure` / `table` 块不参与文本子块的 `0/1/2…` 序列,而是用 mineru manifest 中的 `(page_idx, block_idx)` 作为后缀,形成位置稳定的 `relative_chunk_index`:
>
> ```python
> # 单行多列设计(2026-05 修订):一张图/表 = chunks 表一行,rel_idx 唯一
> rel_idx = f"{chunk_type}:p{page_idx}_b{block_idx}"   # chunk_type ∈ {'table', 'figure'}
> ```
>
> 真实例子:心血管内科学第3版 p343#2 NSTE-ACS 诊断流程图 → `"figure:p343_b2"`。
>
> **碰撞安全**:这类字符串含 `:` `p` `b` 字符,与子块的纯整数 string(`"0"`、`"1"`…)和父块的 `"parent"` 字面互斥,SHA256 输入空间不重叠。同一节里若有多张图/表,每张的 `(page_idx, block_idx)` 由 mineru layout 全书唯一,亦不会互相冲突。
>
> **位置稳定**:`(page_idx, block_idx)` 来自 mineru 原始 layout,POC chunking 参数变化或 enrichment 重跑(prompt 升级、retry-failed)都不会改变 `chunk_id`,保证 ON CONFLICT 覆盖语义。chunk_id 锁定结构位置 vs `content_hash` 锁定文本内容,职责正交,见 §3.1.4.3。
>
> **跨页/多面板合并**:合并组只入库 anchor,sibling 不入库。anchor 的 `(page_idx, block_idx)` 取自 anchor 块自身;sibling 在 manifest 中标 `merge_role="sibling"`,不再独立产生 chunk。anchor 入库时灌 1 行(`chunk_raw_text` 含合并后的 caption + html(table)/ caption(figure),`medical_statement` 由 LLM 看合并截图/合并 html 后生成)。


### 3.1.4.3 content_hash

**作用**：`content_hash` 是变动检测信号字段，与 `chunk_id` 分离，单独存储。

**生成方式**(按 chunk_type 分流)：

```
# child / parent：chunk_raw_text 是唯一文本来源
content_hash = SHA256( chunk_raw_text )

# table / figure：两路文本来源(html/caption 走 BM25,medical_statement 走 dense original)
#                 拼起来哈希,任一变化都触发重 embed
content_hash = SHA256( chunk_raw_text + "\n" + medical_statement )
```

**职责边界**：

| 字段 | 职责 | 是否作为主键 |
|---|---|---|
| `chunk_id` | 结构定位（标题路径 + 块序号），稳定不变 | 是 |
| `content_hash` | 内容变动信号，触发更新 | 否 |

**更新逻辑**：

- Upsert 时，以 `chunk_id` 为主键进行匹配。
- 若 `content_hash` 与数据库中已有值相同 → 跳过 Embedding 计算，复用已有向量（注意：此"跳过"仅针对 Embedding 步骤，chunk_id 的遍历生成始终在全文档范围内完整执行）。
- 若 `content_hash` 不同 → 内容已变更，覆盖写入并重新触发 Embedding 计算。

这样即使文档局部修改，`chunk_id` 保持稳定（结构未变），仅通过 `content_hash` 的差异驱动增量更新，避免全量重建。

**注意：**修改标题时，`chunk_id` 会跟随变化，原标题下的旧 chunk 记录不会被自动覆盖，形成僵尸数据。需在每次文档处理流程中执行以下三步清理：

**文档处理的三步分层逻辑：**

1. **完整遍历（轻量）**：对整篇文档执行完整解析，生成当前版本所有 chunk 的 `chunk_id` 和 `content_hash`，此步骤仅涉及哈希计算，开销极低。
2. **僵尸清理**：以 `source_id` 为范围，从数据库中查出该文档所有已有 `chunk_id`（旧集合），与本次遍历生成的全量 `chunk_id`（新集合）做差集：
   ```
   待删除 = 旧集合 - 新集合   # 在旧集合中存在、但新集合中不存在的记录
   ```
   由于 `parent_chunk_id` 是自引用外键，**必须严格按以下顺序删除**，否则触发 foreign key violation：
   1. 从待删除集合中取出子块（`parent_chunk_id IS NOT NULL`），先删 Milvus 向量，再删 PostgreSQL 记录（走 `idx_chunks_parent_chunk_id` 索引分拣）
   2. 再取出父块（`embedding_status = 'skip'`），无 Milvus 向量，直接删 PostgreSQL 记录
3. **按需重算 Embedding**：对新集合中 `content_hash` 发生变化（或为全新）的**子块**，触发 Embedding 计算；父块（`embedding_status = 'skip'`）始终跳过此步骤；`content_hash` 未变的子块直接复用已有向量。


## 3.1.5 Embedding (多向量化)
在执行的 Embedding 计算之前，计算 Chunk 的内容哈希（Content Hash）。仅针对数据库中不存在的新内容哈希执行向量化计算，对于文件名变更但内容未变的片段，直接复用已有向量，显著降低计算开销。**父块（`embedding_status = 'skip'`）不参与本步骤，直接跳过。**

**混合检索双路架构（Dense + BM25）：**
为了支持高精度的混合检索（Hybrid Search），系统采用两套独立机制：
- Dense Embeddings（语义向量）：调用 Qwen3-Embedding-8B 生成 4096 维浮点向量，捕捉文本的深层语义关联，解决”词不同意同”的检索难题。8B 参数量确保对医学术语的细粒度语义差异有充足的编码能力。
- BM25 全文检索（关键词匹配）：利用 Milvus 2.4+ 内置 BM25 引擎对 `original_content` 建立倒排索引，基于词频/逆文档频率实现精确关键词匹配，解决医学专有名词查找问题。相比 learned sparse（如 SPLADE），传统 BM25 对长尾专业术语更稳定，且可配合医学分词器定制分词。

**多向量表示（文本多向量，Multi-Vector Representation）：**
为进一步提升召回率，系统对每个 Chunk 生成多条向量记录，均指向同一份原始 Chunk 内容。各向量记录携带 `vector_type` 字段加以区分：

| vector_type | 文本来源 | 作用 |
|---|---|---|
| `original` | `child`:`chunk_raw_text`(正文);`table` / `figure`:`medical_statement`(LLM 医学陈述,因 figure chunk_raw_text 只有 caption、table 的 html 作 dense 表达力不足,见 §3.1.2)。`parent` 不参与(`embedding_status='skip'`,仅作 Small-to-Big 展开 context) | 主向量，捕捉原始语义 |
| `summary` | 3.1.3 生成的 `summary` 列 | 摘要向量，提升对模糊 query 的匹配能力 |
| `question` | 3.1.3 生成的 `hypothetical_questions` 列 | 问题向量，弥合患者口语描述与临床文本之间的语义鸿沟 |

每个 Chunk 产出 1 条 `original` + 1 条 `summary` + 3 条 `question` 向量记录，各条记录均通过 `source_chunk_id` 指向原始 Chunk，检索命中补充向量后统一回溯取原始内容（见 3.1.6、3.2.2）。BM25 sparse 始终来自 `chunk_raw_text`(child=正文,table=caption+html+footnote,figure=caption+footnote)。各 vector_type 实际能命中的 query 模式见 §3.1.5.1 字段召回画像。

批处理优化：所有计算均采用 batch_size 驱动的批处理模式，最大化 CPU 利用率并减少网络 RTT。


### 3.1.5.1 字段召回画像（enrichment 字段如何被召回命中）

**适用范围**：本节描述参与多向量召回的 `chunk_type`（`child` / `table` / `figure`）的字段如何被检索命中，是 §3.2 召回策略与 §4 Agent build_query 节点设计 query 改写策略的依据。这些 chunk 共享同一套多向量结构，但 dense `original` 向量来源因 chunk_type 而异：
- `child`：`chunk_raw_text`(正文段落)直接作为 `original` 向量来源；
- `table` / `figure`：`medical_statement`(vision/text LLM 生成的医学陈述,把图表数据线性化为陈述句)作为 `original` 向量来源（替代 chunk_raw_text，因 figure chunk_raw_text 仅含 caption、table 的 html 作 dense 表达力不足,详见 §3.1.2、§3.1.3.2）。

> **不在本节范围**：`parent` 父块（`embedding_status='skip'`，不向量化）。

**字段 → 向量映射**

| 字段 | child 入向量 | table / figure 入向量 | BM25 sparse？ |
|---|---|---|---|
| `chunk_raw_text`（含 title 前缀） | `original` | ✗(不入 dense) | ✓(table=html+caption+footnote,可命中数值/术语;figure=caption+footnote,主要命中"图 N-X"/图名) |
| `medical_statement` | (NULL) | `original` | ✗ |
| `summary` | `summary`（独立向量） | `summary` | ✗ |
| `hypothetical_questions[0..2]` | `question_0` ~ `question_2` | 同左 | ✗ |

> `title` 不单独成向量，作为前缀拼接进 dense `original` 向量来源字段一同入向量(child=chunk_raw_text 前缀,table/figure=medical_statement 前缀)，既参与 dense 也参与 BM25(child) 或仅 dense(table/figure)。

**召回画像表（各字段实际能命中什么样的 query）**

| 字段 | 内容范式 | **高命中** query 模式（✓） | **miss / 弱命中** 模式（✗） |
|---|---|---|---|
| `chunk_raw_text`（child dense `original` / 图表 BM25） | `child`：200-800 字段落叙事；`table`:html+caption+footnote,数据点字面保留;`figure`:caption+footnote(短,通常仅"图 N-X 标题") | child:整体语义贴近的查询;table:含具体术语字面量 / 数值 / 单位 / 罕见专有名（如 `PaCO2 60mmHg 是几级`、`mMRC 3 级具体定义`);figure:含图编号或图名字面（如 `图 2-4-1`、`哮喘发病机制示意图`) | child:主题分散的长查询;table/figure:全口语 0 术语的查询 |
| `medical_statement`(图表 dense `original`) | 200-500 字陈述句堆砌,所有数据点字面保留 | 整体语义贴近的查询;含完整概念名的查询 | 单 cell 数值字面查询(走 BM25 兜底);纯口语症状不带任何术语 |
| `title` | 12-22 字，"实体 + 关系动词 / 限定语"，命名实体高密度 | 短查询（< 15 字）且含本块核心实体（如 `AECOPD 临床分级`、`mMRC 标准`） | 不含本块核心实体名的查询；长口语句子（实体浓度被稀释） |
| `summary` | 100-150 字，叙事化，范式："在 X 情境中，Y 用于 Z；关键点是 ABC" | 方法论 / 关系 / 综述类（如 `COPD 怎么评估`、`X 与 Y 的区别`、`X 是什么 / 用来干嘛`、`胸膜液体交换的压力机制`） | 单 cell 数值查询（summary 不展开 cell）；行级细节（如 `D 组用什么药`，只有 `chunk_raw_text` / `medical_statement` 接得住） |
| `hypothetical_questions` | 3 条，**期望覆盖三种风格**：① 专业术语+疑问；② 症状口语+推断；③ 场景化人物代入 | ① `mMRC 问卷的呼吸困难分级是怎样划分的?`<br>② `走两步路就喘得不行,是不是慢阻肺?`<br>③ `我爸爸有慢阻肺,爬楼喘得厉害,去年犯了两回...` | 同 chunk 内某一行的具体细节问法（3 条覆盖不到所有行）；跨 chunk 综合问法（如 `对比 GCS 和 mMRC`） |

**多向量联合 + Reranker 兜底链路**

```
query
  ├─► question 向量 ×3（主力，患者口语友好）
  ├─► original 向量(child=chunk_raw_text+title 前缀;图表=medical_statement+title 前缀)
  ├─► summary 向量（方法论 / 综述类查询）
  └─► chunk_raw_text 的 BM25 sparse（child=正文 / table=html 数值兜底 / figure=caption 图名兜底）
                ↓
          RRF 联合 → Top-200（§9.7 RETRIEVE_TOP_N）
                ↓
   Reranker 看完整 chunk_raw_text(图表行还可叠 medical_statement)重排 → Top-K
```

**关键认知**：单个 vector 的任务**不是回答 query**，只是**让 chunk 进入 Top-200 候选池**。一旦进池，Reranker 看完整 `chunk_raw_text`(图表行可拼 `medical_statement`) 重排，与命中哪条向量无关。所以 `hypothetical_questions` 不需要枚举所有行 / 所有 cell，只需在某一种风格下让 chunk 露面即可。

**写 query / build_query 节点的实操指引**

| 角色 | 推荐写法 | 命中字段 |
|---|---|---|
| 患者侧（本项目主用例） | 症状口语 + 想知道什么（`走两步就喘是不是慢阻肺急性发作`） | `question` 风格 ② |
| 患者家属 | "我 [人物] 有 [症状] [严重程度]"（`我爸爸喘得厉害爬楼困难`） | `question` 风格 ③ |
| 专业人员 | 术语短句（`mMRC 0~4 级标准`、`AECOPD 分级`） | `title` + `question` 风格 ① + `chunk_raw_text` BM25 |
| Agent build_query 节点 | **同时**生成口语版 + 术语化版两条 query 并行 dense；**sparse 路按维度拆**（`AECOPD Ⅱ级 治疗` 拆成 `[AECOPD][Ⅱ级][治疗]` 三路 BM25） | 风格 ②③ + title + summary + BM25 cell 级兜底，RRF 合并 |

> Agent 节点 ② build_query 必须实现"多视角 query 扩展"与"sparse 维度拆分"，否则患者口语 query 只能命中 `question` 风格 ② / ③，而单 cell / 行级查询的 BM25 兜底失效。详见 §4.x build_query 节点设计。


## 3.1.6 Storage（索引存储）

### 3.1.6.1写入顺序：PostgreSQL 先写，Milvus 后写（串行，不并行）

**设计决策**：采用 **PostgreSQL-first** 串行写入策略，不支持并行写入。

**原因**：
1. **PostgreSQL 是元数据权威源**：Milvus 向量记录通过 `source_chunk_id` 回查 PostgreSQL `chunks` 表获取标题、正文等展示字段（见 2.4.1）。若 Milvus 先写成功但 PostgreSQL 未写入，检索命中的向量将无法关联到元数据，产生**悬挂引用**。
2. **`embedding_status` 天然充当两阶段状态机**：`chunks` 表已有 `embedding_status` 字段（`pending → done → failed`，父块固定为 `skip`），可精确标记 Milvus 写入是否完成，无需额外引入分布式事务。
3. **不并行写入**：两层存储无共享事务协调器，并行写入时任一方失败会产生难以自动修复的不一致状态（一层有数据、另一层没有），排查和补偿逻辑远比串行复杂，收益（节省几十毫秒网络 IO）不值得。

### 3.1.6.2 写入流程（以 Batch 为单位）

```mermaid
flowchart TD
    S1["<b>Step 1: PostgreSQL Upsert（事务）</b><br/>- Upsert chunks 表（chunk_id 为主键）<br/>- 父块：embedding_status = 'skip'<br/>- 子块：embedding_status = 'pending'<br/>- 事务提交"]
    S1 -->|成功| BRANCH
    S1 -->|"失败 → 整批回滚，不进入 Step 2，<br/>下次重试整批重处理"| END1["终止（等待重试）"]

    BRANCH{"父块？<br/>(embedding_status = 'skip')"}
    BRANCH -->|是| END2["终止（无需向量化）"]
    BRANCH -->|否（子块）| S2

    S2["<b>Step 2: Milvus Upsert（批量）</b><br/>- 以 chunk_id 确定性派生向量记录 ID：<br/>&ensp;{chunk_id} (original) / {chunk_id}_summary /<br/>&ensp;{chunk_id}_q0 / {chunk_id}_q1 / ...<br/>- 派生规则与字段定义见 config/milvus_schema.py:119 + src/rag/ingestion/embedding.py:112<br/>- 批量 Upsert 到 docs_collection"]
    S2 -->|成功| S3
    S2 -->|失败| S2a

    S2a["<b>Step 2a: Milvus 写入失败处理</b><br/>- PostgreSQL 中对应 chunk 的 embedding_status 更新为 'failed'<br/>- 记录错误日志（chunk_id + 异常信息）<br/>- 不回滚 PostgreSQL 数据（元数据本身是正确的）<br/>- 由补偿任务定期重试（见下方「补偿机制」）"]

    S3["<b>Step 3: 确认写入</b><br/>- Milvus Upsert 成功后，更新 PostgreSQL：<br/>&ensp;embedding_status = 'done', updated_at = now()"]
```

### 3.1.6.3 失败场景分析

| 失败点 | PostgreSQL 状态 | Milvus 状态 | 系统行为 |
|--------|----------------|-------------|---------|
| Step 1 失败（PG 写入异常） | 无数据（事务已回滚） | 无数据 | 干净状态，直接重试整批 |
| Step 2 失败（Milvus 写入异常） | 有数据，`embedding_status = 'failed'` | 无数据或部分数据 | 元数据正确但无法被检索命中；补偿任务会重试 |
| Step 3 失败（状态回写异常） | 有数据，`embedding_status = 'pending'` | 有数据 | 向量已可检索；补偿任务发现 pending 时会校验 Milvus 侧是否已存在，存在则直接标记 done |

### 3.1.6.4 补偿机制

利用已有的 `embedding_status` 索引（`WHERE embedding_status != 'done'`）定期扫描需要补偿的 chunk：

```python
# 补偿任务伪代码（可作为定时任务或 ingestion 结束后的收尾步骤）
pending_or_failed = db.query(chunks).filter(
    chunks.embedding_status.in_(['pending', 'failed']),  # 显式排除 'skip'（父块），避免死循环
    chunks.updated_at < now() - interval('5 minutes')  # 避免与正在进行的写入冲突
)
for batch in batched(pending_or_failed, size=BATCH_SIZE):
    # 检查 Milvus 侧是否已有对应向量
    existing_ids = milvus.query(ids=[derive_vector_ids(c) for c in batch])

    need_write = [c for c in batch if derive_vector_ids(c) not in existing_ids]
    already_done = [c for c in batch if derive_vector_ids(c) in existing_ids]

    # 补写缺失的向量
    if need_write:
        milvus.upsert(build_vectors(need_write))

    # 统一更新状态为 done
    db.update(already_done + need_write, embedding_status='done')
```

### 3.1.6.5 不回滚 PostgreSQL 的理由

Milvus 写入失败时，**不回滚** PostgreSQL 中已写入的 chunk 元数据：

1. **元数据本身是正确的**：chunk 的文本、标题、摘要等信息与向量化无关，回滚会丢弃有效工作。
2. **回滚后重试成本更高**：若回滚 PG，下次重试需要重新执行 LLM 增强（摘要/问题生成），浪费计算资源。
3. **`embedding_status` 已提供精确的恢复点**：`failed` / `pending` 状态让补偿任务只需处理 Milvus 侧写入，无需重做整条 pipeline。
4. **对用户无感知影响**：`embedding_status != 'done'` 的 chunk 不会出现在检索结果中（Milvus 中无对应向量），不会产生错误的检索体验。

## 3.2 召回策略
采用双路混合检索策略，并行执行稀疏与稠密两条召回路径：
### 3.2.1 内容查询预处理 (Query Processing)

各步骤均在 `build_query` ② 节点内完成，产出 `dense_query`（str）和 `sparse_queries`（list[str]）两个 State 字段，分别作为稠密/稀疏两路的检索输入：

**Sparse Route 专用处理**
1. 关键词识别 (Keyword Extraction):由 ② build_query Step 1 调 LLM 做医学 NER(`build_query_step1_ner`,详见 §9.3),从 `chief_complaint + present_illness` 直接提取实体;**NER 直接产 raw text 进 `confirmed_symptoms` / `denied_symptoms`,不再做 EL 归一化**(EL 整层下线,见 §4.1.6.2);**NER 也不驱动 sparse 词袋**(2026-05-17 RETRIEVAL_EVAL §2 评测决定:中文症状词 EL 50% Tier 3 占位,alias 反查同义词收益低,sparse 改 state 多字段直采)。
2. Sparse 多字段直采(2026-05-17 RETRIEVAL_EVAL §2 改造):`sparse_queries` 由 state 多字段直采,每条作一次独立 BM25 查询(strip 后长度 ≥ 2 过滤、保序去重):
   - **来源 A(state 结构化字段直采)**:
     - `chief_complaint`(顶层主诉)
     - `present_illness_slots` 单值字段:`location` / `duration_pattern` / `onset_mode`(每条独立成袋)
     - `present_illness_slots` list 字段:`associated_symptoms` / `aggravating` / `relieving` / `trigger` / `nature` / `severity`(每个元素独立成袋;2026-05-22:trigger/nature/severity 从单值改 list[str])
   - **来源 B(report 语义信号直入)**:`report_findings` 的 `positive_findings` 每条 + `impressions` 每条作为独立 BM25 词袋(已是医学文献语言,无需 EL 归一化)。这类教材高频鉴别词(如"瞳孔散大"、"右额颞线形骨折")通过 sparse 路直接召回相关 chunk,不再仅靠 Dense LLM 单点综合。
     - **阴性过滤**:`impressions` 中含 `(-)` / `正常` / `阴性` / `未见` / `无异常` 字样的整条跳过(BM25 不懂否定,反向贡献)。`abnormal_values` 原始数值与 `negative_findings` 同样不进 query。
   - **不入 sparse 的字段(理由扎实)**:`present_illness`(200+ 字必然 OR 退化)/ `treatments`(2026-05-22 合并自旧 treatment_tried + treatment_response,半结构化"<药>: <反应>",拉到药学 chunk 不是诊断 chunk + 反应词无 IDF)/ `onset_time`("3 天前" KB 教材不写相对时间)/ `progression`(实测只 3 个泛词)
   - **实测数量**:62 case 平均 21.8 条 sparse(case 001 简单 16 条 / case 062 复杂报告 28 条)
   - **不用 EL alias 反查**(EL 整层下线,terms_collection 数据保留但运行时不查;详见 §4.1.6.2 + EL_DESIGN_REVIEW §11)

**Dense Route 专用处理**
3. Query 整合改写 (Dense Query Construction):LLM 将所有确认症状(`confirmed_symptoms`,EL 移除后为 raw text)、病史关键项、`report_findings` 的 `positive_findings`/`impressions`,以及 `present_illness_slots` 中已填充的维度信息(诱因、加重/缓解因素、症状性质等)整合,改写为语义连贯的自然语言查询句(如"进食后加重的上腹胀痛伴反酸,白细胞升高,既往糖尿病史"),生成单一的 `dense_query`,用于 1 次向量检索。维度信息的纳入使 query 从泛化症状描述细化为具有鉴别特征的临床描述,显著提升召回精度。

### 3.2.2 召回
注意，召回前可以使用元数据提前过滤，缩小候选集、降低成本。

**并行召回 (Parallel Execution)：**
检索范围：两条路径均在 Milvus **全量记录**上执行，不区分 `vector_type`，`original` / `summary` / `question` 三类向量记录均参与召回。

1. **Sparse Route (Milvus BM25)**：以 3.2.1 Step 2 产出的 `sparse_queries` 为输入（来源 A 症状别名词袋 + 来源 B report 语义信号词袋），对每条词袋分别做 Milvus BM25 检索（N 条 = N 次查询），每次返回关键词匹配候选。
2. **Dense Route (Embedding)**：以 3.2.1 Step 3 产出的 `dense_query` 为输入，做 Qwen3-Embedding-8B 编码 -> Milvus ANN 向量检索（Cosine Similarity）-> 返回语义相似候选（1 次检索）。

**结果融合 (Fusion)：单阶段加权多路 RRF**（2026-05-17 RETRIEVAL_EVAL §4 改造)

```
dense_weight   = max(1, N_sparse_actual / RRF_DENSE_WEIGHT_FACTOR)   # 默认 factor=5,见 §9.7
Final_Score(d) = dense_weight · 1/(k + rank_dense(d))                 # dense 加权 1 票
               + Σ 1/(k + rank_sparse_i(d)),                          # sparse 各路等权 1 票
               i ∈ {Sparse 各维度},  k = 60
```

**为什么 dense 要加权(评测改造,RETRIEVAL_EVAL §4)**:
- 旧设计:每路 1 票等权,假设 Sparse 路数 N ≈ 3~5
- §3.2.1 Step 2 改成多字段直采后,**N_sparse 实测涨到 12~30(62 case 均 21.8)**,等权下 dense 单路最大贡献 ≈ 0.0164,sparse 20 路集体命中累加 ≈ 0.236(14× dense),**dense 单路独家命中被严重挤兑**(Top-20 内 dense exclusive chunks 均 0.45 个/case)
- N/5 加权后,D/S 比稳在 1:3~1:4(原 spec 假设的量级),Top-20 内 dense exclusive 保留 ~3 个/case(×6.5)
- factor=5 经 62 case Jaccard 重叠 + chunk type 平衡评测验证,可通过 `.env` `AGENT_RRF_DENSE_WEIGHT_FACTOR=N` 调

**sparse 路自调节特性保留**:某 chunk 命中多个 Sparse 维度时贡献累加,跨模态证据越强权重自动越大。

**dense_weight=1 的退化场景**:当 `N_sparse_actual ≤ RRF_DENSE_WEIGHT_FACTOR`(默认 ≤ 5)时,`max(1, N/5)=1`,行为退化为等权 RRF(向后兼容,符合 spec 原始单元测期望)。

按 RRF 融合分数降序取 Top-N（`settings.agent_limits.RETRIEVE_TOP_N`，初始值 200，**权威定义见 §9.7**；阈值调优改 `.env` 不改代码）截断，丢弃低分长尾候选。

**多向量聚合 (Multi-Vector Aggregation):**
由于 3.1.5 为每个 Chunk 生成了多条向量记录（original / summary / question），同一 Chunk 的不同向量可能同时出现在召回结果中。**Top-N 截断后按 `source_chunk_id` 聚合**:同一 source_chunk_id 下所有命中记录的 `1/(k + rank)` **求和**得到该 chunk 的最终 RRF 分数,**多路命中天然得分更高**(标准 RRF 跨 ranker 求和的同构扩展)。聚合产出的每条候选记录形态:

```python
{
  "source_chunk_id": str,                # PG chunks.chunk_id
  "rrf_score":       float,              # 各 vector 命中分数之和
  "vector_hits": [                       # 命中清单(每命中一种 vector_type 加一条)
    {
      "vector_type":  str,               # original / summary / question
      "rank":         int,               # 在原召回路中的排名
      "matched_text": str,               # 该向量对应的源文本(取值规则见下)
    },
    # ...
  ],
}
```

**matched_text 取值规则**:
- `vector_type='original'`:取 `chunks.chunk_raw_text`(等价 Milvus payload `original_content`)
- `vector_type='summary'`:取 `chunks.summary`
- `vector_type='question'`:从 Milvus vector ID(`{chunk_id}_q{n}`,见 §3.1.6.2 + config/milvus_schema.py:119)解出 n,取 `chunks.hypothetical_questions[n]`

聚合后按 `rrf_score` 降序取 Top-K,**Top-K 仍为 K 个唯一 source_chunk_id**(结构去重不变),但 `vector_hits` 副载荷保留多路命中证据,供 §3.2.3 Context 扩展按命中向量类型决定附加上下文(命中文本回带 + 反查链)。

设计动机见 §3.2.3 规则 4(命中向量文本回带)。

### 3.2.3 精确过滤与重排

**Metadata Filtering Strategy（元数据过滤策略）**
核心原则：**能前置则前置，无法前置则后置兜底**。

- **解析**：Query Processing 阶段将结构化约束解析为通用 filters（如 collection / doc_type / language / time_range / access_level 等）。
- **Pre-filter（硬约束）**：若底层索引支持，在 Dense/Sparse 检索阶段提前过滤，缩小候选集、降低成本。
- **Post-filter（兜底）**：索引不支持或字段质量不稳的过滤，在 Rerank 前统一执行；字段缺失时默认"宽松包含"（missing → include），避免误杀召回。
- **软偏好（Soft Preference）**：如"更近期更好"，不做硬过滤，作为排序信号在融合/重排阶段加权处理。
---
**Rerank Backend（可插拔精排后端 — diagnose ⑩ 前置，非检索阶段）**

Cross-Encoder 精排**不在 `retrieve` ③ 中调用**。检索阶段（`retrieve` ③）首轮召回量大（~500 chunks），不适合 Cross-Encoder 逐对打分。Reranker 在 `diagnose` ⑩ 前置调用——经过多轮追问/检查后候选集已收敛至可控规模，此时对 `candidate_chunks` 做 Cross-Encoder 精排截断至 Top-K，再交由 LLM 做临床决策排序。

该模块**必须可关闭**，并提供稳定回退策略。

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **None(关闭,默认)** | 跳过 Cross-Encoder,直接将 `candidate_chunks[:RERANK_TOP_K]` 原序传入 LLM 诊断 | 默认模式 — 2026-05-17 RETRIEVAL_EVAL §7 评测验证 |
| Cross-Encoder | 输入 [Query, Chunk] 对，输出相关性分数排序，截断至 Top-K | 备选 — 评测显示 K=20 下全主指标无优势(Hit -1.6pp / NDCG -0.076 / MRR -0.065) |

**默认策略(2026-05-17 RETRIEVAL_EVAL §7 评测确定)**:
- `RerankerSettings.ENABLED = False`(`config/settings.py`),`.env` 用 `RERANKER_ENABLED=True` 覆盖开启
- 评测依据:全 62 case 在 K=5 / K=10 / K=20 / K=50 所有 K 段,BGE-Reranker-v2-minicpm-layerwise 在所有主指标(Hit@K / NDCG@K / MRR(≥2))上均无优势,Reranker NDCG@20 win 12/62 (19%),MRR(≥2) win 2/62 (3%)
- 收益:省 2.6 GB GPU 显存 + ~5 秒/次延迟,NDCG@20 反而提升 +0.076
- 兼容:Cross-Encoder 不可用、超时或失败时,**必须回退至 `candidate_chunks` 原序**,确保 `diagnose` ⑩ 可正常执行

---
**Context 扩展(Context Expansion — diagnose ⑩ 内置,非独立检索步骤)**

Cross-Encoder 精排截断出 Top-K chunk 后,在构建 LLM prompt 前对每个 Top-K chunk 按其 `chunk_type` 应用对应扩展规则,把"召回锚点"展开为"完整上下文":

**规则 1:子块 → 父块**(Small-to-Big,见 §3.1.2):
- 命中 `chunk_type='child'` 的小块,按 `parent_chunk_id` 批量回查 PostgreSQL,取父块全文(`chunk_raw_text`)
- 若 `parent_chunk_id IS NULL`,保留小块原文兜底

**规则 2:图表 chunk → 父块**(章节上下文,见 §3.1.2):
- 命中 `chunk_type ∈ {table, figure}` 的图表 chunk,其 `chunk_raw_text`(table=html+caption+footnote / figure=caption+footnote)+ `image_path` 已在同一行直读可得
- **按 `parent_chunk_id` 回查所在节父块,把父块全文合并进 context**——保证图表命中后 LLM 能看到所属章节的临床上下文(否则只看图本身可能不知道是哪个病的什么阶段)
- 进入 ⑩ Step 1 时按 `image_path` 加载 figure 截图作为视觉输入(详见本节末"LLM 路由"段);**`medical_statement` 仅作召回辅助(让图表 chunk 进 dense Top-N 池),不进 prompt**(见 §3.1.5.1 关键认知段)

**规则 3:父块 → 同节图表**(图表跟随父块,见 §3.1.2):
- 规则 1 展开出的父块或 Top-K 直接命中的父块,按 `heading_path_id` 批量查同节内的图表 chunk(`chunk_type ∈ {table, figure}`)
- 全部合并进 LLM context,**封顶 `settings.agent_limits.RETRIEVE_PARENT_FIGURE_CAP`**(默认 5,见 §9.7)
- 超过封顶按 `relative_chunk_index` 升序保留前 K 个(确定性顺序,无优先级判断)
- 每条图表合并时把 `chunk_raw_text`(table=html / figure=caption + footnote)+ `image_path` 截图一并送进 ⑩ Step 1 prompt;`medical_statement` 不进 prompt(理由同规则 2)

**规则 4:命中向量文本回带**(连接 §3.2.2 vector_hits 副载荷):
- §3.2.2 多向量聚合产出的每条候选 chunk 携带 `vector_hits` 清单(命中的 vector_type / rank / matched_text)
- Context 扩展时,**除规则 1~3 展开的父块/图表 chunk 外,把 vector_hits 中所有 matched_text 也作为 prompt hint 附加**(去重:matched_text 已被父块原文覆盖的整体跳过);其中 vector_type='question'/'summary' 的 matched_text 来自 enrichment LLM 生成,**仅承担召回信号回带语义线索**的职责,不视为权威医学事实
- 设计动机:summary / question 文本在 enrichment(§3.1.3)时由 LLM 看着完整 `heading_path` 生成,**天然带有标题路径上下文**(如"肺炎的临床表现")。父块原文可能 standalone 看不出病名(只是"三、临床表现"几段),把命中的 summary/question 文本送进 prompt 让 LLM 能看到"为什么这条 chunk 被召回"的语义线索
- **Schema 影响**:`candidate_chunks: list[dict]` 子结构按 §3.2.2 形态,Pydantic 不强约束 list 元素 schema,新增 vector_hits 字段属 §9.2 第三类(给字段加默认值即可,反序列化老 state 时缺 vector_hits 视为 [],兼容)

**去重**:四条规则展开后按 chunk_id 去重(常见 case:图表 chunk 直接命中触发规则 2,父块展开触发规则 3,同一图表被两条路径都拉出来;规则 4 的 matched_text 仅在不与已展开正文重叠时附加)。

**作用域**:扩展产物**仅用于当次 LLM prompt 构建**,不写回 `candidate_chunks` State 字段。`candidate_chunks` 全程存储 Top-K 原 chunk,其余节点对扩展逻辑完全无感知。

**为什么图表 chunk 同行多列**(2026-05-12 重构):原"源 + summary 两行"架构需 `linked_chunk_id` 回查,新单行设计下 `chunk_raw_text`(table=html / figure=caption)+ `medical_statement`(LLM 陈述)+ `image_path`(截图)同行直读,LLM 一次拿到全部结构化内容,无 JOIN。

**为什么父块要带同节图表**:图表 chunk 在 Top-M 抢位上可能输给正文 chunk(Reranker 偏好自然语言 chunk)。但只要同节任一正文 child 命中 → 规则 1 拉父块 → 规则 3 拉同节图表,**图表跟随正文召回的兜底路径**因此打通,不依赖图表 chunk 自身进 Top-K。

**父块大小**(2026-05-03 POC 验证):新切分方案下父块 median 1346 字 / p95 3563 字 / max 5218 字(~720~3700 token),约 56% 父块 > 1200 字会切多 child(其余 44% 父块 ≤ 1200 字,1 child = parent 整段)。父块全文 + 5 个图表的 payload 塞入 LLM prompt 完全可控,**不需要做任何"展开整节为多 chunk"的额外扩展逻辑** — 直接用父块文本 + 图表 payload 即可。

**LLM 路由**(⑩ diagnose 1 步 LLM,原生多模态):

⑩ 现在是 1 步 LLM 走 DashScope qwen3.5-plus(`settings.llm.VISION_*`,见 §9.3)。context 中 figure chunk 的 `image_path` 转 base64 作为多模态消息送入,LLM 自己看图;table chunk 仅送 `chunk_raw_text`(html 已是高质量文本);figure 不在 context 时也固定走 vision LLM,代码无分支判断。

**为什么固定走 vision 而不按需切换**:
- 代码层避免"看 context 里有没有 figure 再决定走哪个 LLM"的运行时分支判断,提高可读性与可测性
- vision LLM 处理纯文本输入也没问题,只是成本略高;⑩ 是 Agent 主路径核心节点,稳定性优于成本
- enrichment 阶段的图表入库(C4 figure_enrichment 已走 vision LLM 看图生成 `medical_statement` 用于召回)与诊断阶段的"再看一次图"职责分离 — 召回阶段的 vision LLM 看到的是孤立图,诊断阶段的 vision LLM 看到的是图 + 完整章节上下文 + 患者主诉 + 多轮证据,语义工作面完全不同,二者结论可能冲突 → 以诊断阶段为准


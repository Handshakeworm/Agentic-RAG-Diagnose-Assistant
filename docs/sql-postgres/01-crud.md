# 增删查改(CRUD)+ UPSERT —— 地基层学习笔记 ①

> 本文件夹(`docs/sql-postgres/`)专门放**地基层 SQL + PostgreSQL** 逐点笔记,配简要例子。
> 总览见 [../DATABASE_LEARNING_ROADMAP.md](../DATABASE_LEARNING_ROADMAP.md) 的 🥇 地基层。
> **每个关键概念都标英文术语(English term)**,方便英文语境面试/实习表达。每个操作下用 `⬜ 还没学` 列出尚未掌握的用法(一个知识点一行 + 英文)。

## 目录
1. [先建立画面:表 = Excel](#1-先建立画面表--excel)
2. [增 INSERT(Create)](#2-增-insertcreate)
3. [查 SELECT(Read)](#3-查-selectread)
4. [改 UPDATE(Update)](#4-改-updateupdate)
5. [删 DELETE(Delete)](#5-删-deletedelete)
6. [四个一起记](#6-四个一起记)
7. [UPSERT(增的进阶)](#7-upsert增的进阶)
8. [吞掉冲突的场景](#8-吞掉冲突的场景)
9. [UPSERT 为什么并发安全(深入)](#9-upsert-为什么并发安全深入)
10. [主键 vs 唯一约束](#10-主键-vs-唯一约束)
11. [易错点速查](#11-易错点速查)

---

## 1. 先建立画面:表 = Excel

```
patients 表(table)
┌────────┬────────┬────────┬───────────┐
│ id     │ name   │ gender │ height_cm │   ← 列(column)= 属性
├────────┼────────┼────────┼───────────┤
│ p_001  │ 张三   │ male   │ 175       │   ← 行(row)= 一条记录(record)
└────────┴────────┴────────┴───────────┘
```

- 一行(row)= 一条记录(record);一列(column)= 一个属性(attribute)。
- **增删查改(CRUD)= 对表做的四件事**:加行 / 删行 / 看行 / 改行。
- 学习顺序按"一条数据的一生(lifecycle of a row)":**增 → 查 → 改 → 删**。

| 中文 | 关键字 | English |
|---|---|---|
| 增 | `INSERT` | Create |
| 查 | `SELECT` | Read |
| 改 | `UPDATE` | Update |
| 删 | `DELETE` | Delete |

> SQL **对空白不敏感(whitespace-insensitive)**:换行/缩进只为可读性(readability),语句以分号 `;` 结束。短就一行,长就每个子句(clause)一行。

---

## 2. 增 INSERT(Create)

**作用**:塞一条新记录(insert a new row)。

### 2.1 基本插入(basic insert)

```sql
-- 单行(single row)
INSERT INTO patients (id, name, gender, height_cm)
VALUES ('p_001', '张三', 'male', 175);

-- 多行插入(multi-row insert,一条语句插多行,比循环快)
INSERT INTO patients (id, name, gender) VALUES
  ('p_002', '李四', 'female'),
  ('p_003', '王五', 'male');
```

**记住**:括号里的**列(columns)**与后面的**值(values)**一一对应、顺序一致。

### 2.2 `INSERT ... SELECT`(用查询结果当数据源)✅

把 `VALUES (...)` 整块**换成一条查询(query)**,要插入的行**来自 SELECT 的结果**,而不是手写字面值。`VALUES` 和 `SELECT` **二选一**。

```sql
-- 把所有男性病人复制进归档表(SELECT 查出多少行就插多少行)
INSERT INTO patients_archive (id, name, status)
SELECT id, name, 'archived'      -- 'archived' 是写死的常量(literal)
FROM patients
WHERE gender = 'male';
```

- SELECT 输出列与目标列要**一一对应**(数量/顺序/类型)。
- 主要用途:**批量搬数据**——归档(archive)、回填(backfill)、派生表。

### 2.3 自增主键(auto-increment primary key)✅

让数据库**自动生成递增的唯一 id**,插入时不用自己给。

```sql
-- 现代标准写法(推荐):GENERATED ... AS IDENTITY
CREATE TABLE t (
  id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  name text
);
INSERT INTO t (name) VALUES ('张三');   -- 不写 id,DB 自动填 1,2,3...
INSERT INTO t (name) VALUES ('李四') RETURNING id;   -- 拿回生成的 id
-- 老写法(legacy):id SERIAL PRIMARY KEY
```

- 底层是**序列(sequence)**:专门"发号"的对象,原子地吐下一个号(`nextval`)。
- ⚠️ **会有空洞(gaps)**:回滚/并发会消耗号但不回填 → id **唯一且递增,但不保证连续**(面试常考)。
- 术语:为做主键而人造的列叫**代理键(surrogate key)**;用业务已有唯一值当主键叫**自然键(natural key)**。
- 🟡 **了解即可**:`SERIAL` 是老写法,新表用 `IDENTITY` 就行;其他库的叫法(MySQL `AUTO_INCREMENT` · SQL Server `IDENTITY` · SQLite `AUTOINCREMENT`)认得即可,不必练。

#### UUID vs 自增整数(区别 + 选型 / when to choose)

两者都是**代理键(surrogate key)**——人造、无业务含义的主键。根本区别:

| 维度 | 自增整数 auto-increment | UUID |
|---|---|---|
| 怎么保唯一 | 一个**计数器**按序发号 | **随机** + 空间巨大,几乎不撞 |
| 谁能生成 | 只能**中心**(DB 序列)发 | **谁都能**自己生成,零协调 |
| 样子 | `1, 2, 3 …` | `550e8400-e29b-…` 随机串 |
| 有序? | 递增有序 | 无序(v4) |
| 暴露信息? | 会(顺序/数量可被 `+1` 猜) | 不会(随机) |
| 大小/索引 | 小、索引友好 | 大(16 字节)、随机值对索引不友好 |

**一句话根因**:自增靠计数器 → **必须有一个中心发号**;UUID 靠随机 → **零协调,哪都能生成**。下面所有差异都从这句长出来。

**选型(默认自增,命中任一就上 UUID)**:
- id 对外暴露(URL / API),不想被 `+1` 枚举出别人数据 → **UUID**(隐私 / 安全)
- 多节点 / 多库各自生成、将来分库 → **UUID**(不撞、无单点瓶颈)
- 跨环境 / 多来源合并数据 → **UUID**
- 否则(单库、内部、id 不外露)→ **自增整数**(小、快、简单)

> 🟡 **了解即可(关键在"谁生成 id")**:UUID 的"重试幂等""父子引用提前接好"两个好处,**只有客户端生成 id 时才有**(id 入库前就已知);**服务端生成的 UUID 享受不到**。服务端生成下想要接口幂等,用独立的 **`Idempotency-Key`**。

> 📌 **你项目**:[../../src/db/postgres/models_patient.py](../../src/db/postgres/models_patient.py) 等用 `PG_UUID` 主键 + `server_default=func.gen_random_uuid()` → **数据库端生成 UUID**(不是应用生成)。选 UUID 为了:**① 不可枚举(医疗隐私)② 去单点发号 / 分布式友好 ③ 合并不撞**。(知识库 `sources` / `chunks` 用语义 / 内容键,是另一种考量。)

### 2.4 RETURNING(插/改/删完拿回数据)✅

`INSERT` / `UPDATE` / `DELETE` **默认不返回数据**(只报影响了几行)。加 `RETURNING` 让它**返回受影响行的某些列**。

```sql
INSERT INTO patients (name) VALUES ('张三')
RETURNING id, created_at;     -- 插入并立刻拿回 DB 生成的 id、创建时间
```

最常用途:**拿回数据库生成的值**(尤其 id)。这正是前面"插完要读回 id"的那个**读回**——`RETURNING id` 在**同一条语句**里完成,不用再单独 `SELECT`(少一次往返,也避免并发下读错)。也能用在改/删:

```sql
UPDATE patients SET height_cm = 178 WHERE id = 'p_001' RETURNING height_cm;
DELETE FROM patients WHERE id = 'p_001' RETURNING *;   -- 返回被删整行(留底/确认)
```

**能返回什么**:受影响行的**任意字段**(含你没写的、DB 自动生成的列,或 `*` 全部)、**字段上的表达式/函数**(可加别名 `AS`);**每影响一行返回一行**(像查询结果集),内容是操作后的状态(改后新值 / 被删的那行)。

> 🟡 了解即可:`RETURNING` 是 PostgreSQL(等)的特性;MySQL 传统没有,改用 `LAST_INSERT_ID()` 之类。

**⬜ 还没学(增)—— 均 🟡 了解即可,记一句"是什么"就够**

- 🟡 `INSERT ... DEFAULT VALUES` —— 插一行,所有列都用各自默认值(连 `VALUES` 都不写),罕用
- 🟡 `COPY` —— PG 的批量高速导入(从文件/流一次灌大量行),比逐条 INSERT 快几个数量级;数据加载/ETL 用,应用代码很少手写
- 🟡 `MERGE`(PG 15+)—— SQL 标准"合并",一条语句按匹配与否做 INSERT/UPDATE/DELETE 多动作;比 `ON CONFLICT` 通用,但 PG 日常 upsert 用 `ON CONFLICT` 就够
- (`ON CONFLICT` 即 UPSERT、`INSERT ... SELECT`、自增主键、`RETURNING` 均已学 ✅)

---

## 3. 查 SELECT(Read)

**作用**:读数据(query),**只看不改(read-only)**。

### 3.1 骨架(skeleton)

```sql
SELECT 要看的列  FROM 表名  WHERE 条件;

SELECT * FROM patients;                          -- 全部行全部列
SELECT name, height_cm FROM patients;            -- 只看某几列(projection)
SELECT * FROM patients WHERE gender = 'male';     -- 加条件(filter)只看男性
```

**记住**:不写 `WHERE` = 返回所有行;写了 = 只返回符合条件的行。

### 3.2 WHERE 条件家族(怎么写筛选条件)✅

`WHERE` 后面的"条件"决定**留下哪些行**。六类写法:

**比较运算符 / comparison operators**:`=` `<>` `>` `<` `>=` `<=`
```sql
SELECT * FROM patients WHERE height_cm > 170;
SELECT * FROM patients WHERE gender <> 'male';   -- <> 是标准"不等于"(!= 也行)
```

**逻辑组合 / logical operators**:`AND` `OR` `NOT`
```sql
SELECT * FROM patients WHERE gender = 'male' AND height_cm > 170;
```
⚠️ `AND` 优先级高于 `OR`,混用**加括号**:`WHERE (a OR b) AND c`。

**模糊匹配 / pattern matching**:`LIKE` + 通配符 `%`(任意多个字符)`_`(恰好一个字符)
```sql
SELECT * FROM patients WHERE name LIKE '张%';   -- 姓张
SELECT * FROM patients WHERE name LIKE '_娟';   -- 两字、第二字是"娟"
```
🟡 `ILIKE` = 忽略大小写版(PG 特性)。

**集合判断 / membership**:`IN` / `NOT IN`
```sql
SELECT * FROM patients WHERE gender IN ('male', 'female');   -- 一堆 OR 的简写
```

**范围判断 / range**:`BETWEEN ... AND`(**含两端**)
```sql
SELECT * FROM patients WHERE height_cm BETWEEN 160 AND 180;   -- 160 ≤ x ≤ 180
```

**空值判断 / null check**:`IS NULL` / `IS NOT NULL`
```sql
SELECT * FROM patients WHERE birth_date IS NULL;   -- ⚠️ 不能写 = NULL(老坑)
```

**自由组合**:
```sql
SELECT name, height_cm FROM patients
WHERE gender = 'male'
  AND height_cm BETWEEN 170 AND 185
  AND name LIKE '王%';
```

> 一句话:`WHERE` 用这六类(比较 / 逻辑 / 模糊 / 集合 / 范围 / 空值)拼条件,可任意 `AND`/`OR` 组合。

### 3.3 排序 ORDER BY(sorting)✅

把结果按某列排序。默认升序 `ASC`(从小到大),降序写 `DESC`。

```sql
SELECT name, height_cm FROM patients ORDER BY height_cm DESC;   -- 按身高从高到矮
SELECT * FROM patients ORDER BY gender ASC, height_cm DESC;      -- 先按性别分,同性别内再按身高降序
SELECT * FROM patients ORDER BY birth_date DESC NULLS LAST;      -- 生日新→旧,没填生日的(NULL)排最后
```

- **多列排序**:逗号分隔,前面的列优先,前列相同才看后列。
- **`NULLS FIRST|LAST`**:控制 `NULL` 排最前还是最后(PG 默认 `ASC` 时 NULL 在最后、`DESC` 时在最前)。
- 🟡 可按 `SELECT` 里没出现的列排序,也能按列序号排(`ORDER BY 2` = 按第 2 列);序号写法可读性差,知道即可。
- **面试钩子**:排序要花 sort 成本;若排序列上有索引(index),数据库可**直接按索引顺序读、免排序**——这是给 `ORDER BY` 建索引的动机。

### 3.4 分页限量 LIMIT / OFFSET(pagination)✅

`LIMIT N` 只取前 N 条;`OFFSET M` 先跳过 M 条。两者合用就是"翻页"。

```sql
SELECT * FROM patients ORDER BY id LIMIT 10;             -- 前 10 条
SELECT * FROM patients ORDER BY id LIMIT 10 OFFSET 20;   -- 跳过 20 条 → 取第 21~30 条(每页 10 条的第 3 页)
```

- ⚠️ **分页必须配 `ORDER BY`**:不排序时行的返回顺序不保证,翻页会出现重复/漏行。
- 🟡 SQL 标准等价写法 `FETCH FIRST 10 ROWS ONLY`,了解即可,PG 日常用 `LIMIT`。
- **面试钩子——深翻页为什么慢(deep pagination)**:`OFFSET 100000` 数据库仍要**先扫描并丢弃前 10 万行**,翻得越深越慢。解法是 **keyset pagination(游标翻页)**:记住上一页最后一行的 id,用 `WHERE` 直接跳过,代价与页深无关。

```sql
-- ❌ 深翻页:OFFSET 越大越慢(要扫描+丢弃前面所有行)
SELECT * FROM patients ORDER BY id LIMIT 10 OFFSET 100000;
-- ✅ keyset pagination:记住上一页最后的 id,直接定位
SELECT * FROM patients WHERE id > '上一页最后一行的id' ORDER BY id LIMIT 10;
```

### 3.5 去重 DISTINCT(deduplication)✅

`SELECT DISTINCT` 把**完全重复的行**只留一份。

```sql
SELECT DISTINCT gender FROM patients;             -- 一共有哪些不同性别
SELECT DISTINCT gender, height_cm FROM patients;  -- ⚠️ 是(性别,身高)组合去重,不是只看性别
```

- ⚠️ `DISTINCT` 作用于 `SELECT` 列出的**所有列的组合**(整行),不是只对第一列去重——这是常见误解。
- 🟡 **`DISTINCT ON (列)`**(PG 专有):每个该列值只取一行,配 `ORDER BY` 决定取哪行。例:每个性别取身高最高的人。了解即可。

```sql
SELECT DISTINCT ON (gender) gender, name, height_cm
FROM patients ORDER BY gender, height_cm DESC;    -- 每个性别只保留身高最高的那行
```

- **面试钩子**:`DISTINCT` 和 `GROUP BY` 都能去重;聚合时常用 `COUNT(DISTINCT gender)` 数"有几种不同性别"。

### 3.6 别名 AS(alias)✅

给列或表起个临时名字,只在这条查询里有效。

```sql
SELECT name AS 姓名, height_cm AS height_m_source FROM patients;  -- 列别名:结果表头更友好
SELECT p.name, p.height_cm FROM patients AS p;                   -- 表别名 p:多表时区分同名列、少打字
```

- 列别名 / 表别名的 `AS` 都**可省略**(`height_cm height`、`patients p`),但写上更清楚。
- **面试钩子——逻辑执行顺序(logical query processing order)**:`FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT`。因为 `SELECT` 在 `WHERE` **之后**才执行,所以 **`WHERE` 里用不了 `SELECT` 起的别名**,而 `ORDER BY`(最后执行)可以用。

```sql
-- ❌ WHERE 比 SELECT 先执行,别名 h 还不存在:
-- SELECT height_cm AS h FROM patients WHERE h > 170;
SELECT height_cm AS h FROM patients ORDER BY h DESC;   -- ✅ ORDER BY 最后执行,能用别名
```

### 3.7 表达式与函数 expressions & functions ✅

`SELECT` 里不只能取原始列,还能**现算现转**,生成派生列(derived column)。

```sql
-- 算术:身高 cm 换算成 m
SELECT name, height_cm / 100.0 AS height_m FROM patients;

-- 字符串函数:转小写、取长度、拼接
SELECT lower(name), length(name), concat(name, '-', gender) FROM patients;

-- 日期函数:从生日算年龄
SELECT name, extract(year FROM age(birth_date)) AS age_years FROM patients;

-- CASE WHEN:类似 if-else,把值映射/分桶(bucketing)
SELECT name,
       CASE WHEN height_cm >= 180 THEN '高'
            WHEN height_cm >= 165 THEN '中'
            ELSE '矮' END AS height_level
FROM patients;

-- COALESCE:空值兜底,第一个非 NULL 的值;没填身高就显示 0
SELECT name, COALESCE(height_cm, 0) AS height_or_zero FROM patients;
```

- **`CASE WHEN ... THEN ... ELSE ... END`**:SQL 里的条件分支,等于编程里的 if-else,常用来做派生列 / 分桶。
- **`COALESCE(a, b, ...)`**:返回第一个非 `NULL` 的值,是处理空值最常用的函数(对照 §3.2 的 `IS NULL`:一个判断空、一个兜底空)。
- **面试钩子**:把"展示/计算逻辑"下推到 SQL(数据库现算)还是放到应用层算,是常见取舍——简单转换交给 SQL 省传输,复杂业务逻辑放应用层好维护。

### 辨析:三种"合并"别混(JOIN / 集合运算 / UPSERT·MERGE)

中文里 JOIN、UNION、MERGE 都能叫"合并",但分属**两侧、三种动作**。看清这张图就不会混:

**读侧(出结果集,不改表):**

| 动作 | 中文 | 干什么 | 形象 |
|---|---|---|---|
| `JOIN` | 连接 | 按匹配条件把多表的**列**拼成一张宽表 | 横向加宽——**拼列** |
| `UNION`/`INTERSECT`/`EXCEPT` | 集合运算 / set operations | 把两个结果集的**行**纵向叠 / 取交 / 取差 | 纵向堆高——**叠行** |

**写侧(改目标表):**

| 动作 | 中文 | 干什么 |
|---|---|---|
| `INSERT ... ON CONFLICT`(UPSERT) | 插入或更新 | 主键 / 唯一键撞了就改、没撞就插 |
| `MERGE`(PG 15+) | 归并写入 | 一条语句按匹配与否做 INSERT/UPDATE/DELETE,是 UPSERT 的超集 |

一句话:
- **JOIN ≠ MERGE**——一个是读(拼列看),一个是写(灌数据);它俩不是一对,别放一起记。
- JOIN 的"同类"是**集合运算**(都是读侧 combine,只是 JOIN 拼列、UNION 叠行)。
- MERGE 的"同类"是 **UPSERT**(都是写侧 combine,MERGE 更通用,PG 日常 `ON CONFLICT` 就够)。

JOIN 自己的类型细分留到下面「多表关联」展开:`INNER`(只留匹配上的)/ `LEFT`(左表全留,右边没匹配补 `NULL`)/ `RIGHT` / `FULL` / `CROSS`(笛卡尔积)。

**⬜ 还没学(查,后续一个一个展开)**

**A. 单表查询基础 —— ✅ 已学(见上 §3.3–§3.7:ORDER BY / LIMIT·OFFSET / DISTINCT / AS / 表达式与函数)**

**B. 多表 / 汇总(进阶必学)**
- 🔴 多表关联 / join:`JOIN` —— 按匹配把多表的列拼成宽表(`INNER`/`LEFT` 最常用)
- 🔴 聚合 + 分组 / aggregation:`GROUP BY` / `HAVING` —— 按某列分组算汇总(`COUNT`/`SUM`/`AVG`/`MAX`/`MIN`);`HAVING` = 分组后再筛
- 🔴 集合操作 / set operations:`UNION` / `UNION ALL` / `INTERSECT` / `EXCEPT` —— 两个结果集纵向叠 / 取交 / 取差
- 🔴 子查询 / subquery:`IN` / `EXISTS` / 标量子查询 —— 查询里套查询

**C. 进阶(按需,不平均用力)**
- 🟡 窗口函数 / window functions:`OVER (PARTITION BY ...)` —— 分组算汇总但**不合并行**(排名、累计);知道有这回事即可
- 🟢 公共表表达式 / CTE:`WITH ... AS` —— 给子查询命名、把复杂查询拆成几步;顺带就会
- 🟡 全文检索 / full-text search:`tsvector` / `@@` —— PG 自带文本搜索;本项目交给向量库 BM25,可跳过

---

## 4. 改 UPDATE(Update)

**作用**:改已有记录的某些列(modify existing rows)。

```sql
UPDATE patients
SET height_cm = 178
WHERE id = 'p_001';
```

**记住(命根子)**:`UPDATE` 不写 `WHERE` → **全表每行都被改**。
改几行**由 WHERE 决定**:`WHERE id = 'p_001'`(唯一)改一行;`WHERE gender = 'male'`(非唯一)改一批。多行是正常能力,危险在于条件太松误伤。
**安全习惯**:改之前先用同样 WHERE 跑 `SELECT` 看命中几行(verify the row count first)。

**⬜ 还没学(改)**

- `UPDATE ... FROM` —— 用另一张表的数据来更新本表 / update-from-another-table(join 式更新)
- `UPDATE ... WHERE (子查询)` —— 更新条件来自一条子查询 / subquery in WHERE
- 一次 `SET` 多列 + 用表达式 / multi-column & expression update(`SET n = n + 1`,UPSERT 计数那讲过)
- (`RETURNING` 见 §6)

---

## 5. 删 DELETE(Delete)

**作用**:删掉行(remove rows)。

```sql
DELETE FROM patients
WHERE id = 'p_001';
```

**记住(命根子)**:`DELETE` 不写 `WHERE` → **删光整张表**。

两个进阶概念:
- **软删除(soft delete)vs 物理删除(hard delete)**:不真删,而是标记一列(如 `status` / `deleted_at`),查询时 `WHERE deleted_at IS NULL` 过滤。好处:可恢复、留审计(auditable)。
- **`DELETE` vs `TRUNCATE`**:`DELETE` 逐行删、可带 `WHERE`、可回滚(rollback);`TRUNCATE` 清空整表、极快、不能带条件、几乎不可回滚。"清空重灌"用 `TRUNCATE`,"删符合条件的几行"用 `DELETE`。

**⬜ 还没学(删)**

- `DELETE ... USING` —— 基于另一张表的条件来删 / delete-using-another-table
- `DELETE ... WHERE (子查询)` —— 删除条件来自一条子查询 / subquery in WHERE
- `ON DELETE CASCADE` —— 外键级联删除 / cascading delete(删父行自动删子行;属外键约束)
- (`RETURNING` 见 §6;`TRUNCATE` 已学)

---

## 6. 四个一起记

| 操作 | 关键字 | 骨架 | 干啥 |
|---|---|---|---|
| 增 Create | `INSERT INTO` | `INSERT INTO 表 (列) VALUES (值)` | 加行 |
| 查 Read | `SELECT` | `SELECT 列 FROM 表 WHERE 条件` | 看行(不改数据) |
| 改 Update | `UPDATE` | `UPDATE 表 SET 列=值 WHERE 条件` | 改行 |
| 删 Delete | `DELETE` | `DELETE FROM 表 WHERE 条件` | 删行 |

**两句话锁死**:
1. **只有"查(Read)"不改变数据**,增/改/删都会动表(mutate)。
2. **`WHERE` 决定"对哪些行动手"**——改和删忘了 `WHERE` 就是对全表动手。

**⬜ 还没学(跨增改删通用 / cross-cutting)**

- `RETURNING` —— 增/改/删都能返回"受影响的行"(PG 常用)。**已学 ✅ 见 §2.4**
- 参数化查询 / 预编译语句 / parameterized (prepared) statement —— 从代码调 SQL 时防 **SQL 注入(SQL injection)**,应用开发必会
- 事务 / transaction:`BEGIN` / `COMMIT` / `ROLLBACK`(独立大主题,见 [../DATABASE_LEARNING_ROADMAP.md](../DATABASE_LEARNING_ROADMAP.md))
- 索引 / index(独立大主题,见路线图)

---

## 7. UPSERT(增的进阶)

**定性**:UPSERT 不是第五个操作,是一种**写入策略(write strategy)= INSERT 为主 + 冲突了就退化成 UPDATE**(**UP**date + in**SERT**)。

```sql
INSERT INTO patients (id, name, height_cm)
VALUES ('p_001', '张三', 178)
ON CONFLICT (id)                              -- 看 id 这个唯一键冲不冲突
DO UPDATE SET height_cm = EXCLUDED.height_cm; -- 冲突就改成更新
```

**运行逻辑(两种结局)**:
1. 先尝试插入这行。
2. 检查是否撞上**唯一约束/主键(unique constraint / primary key)**(表里已有同 key 的行?)。
3. 结局:
   - **没撞** → 正常插入新行。
   - **撞了(conflict)** → **不报错**,改执行 `DO UPDATE`(更新旧行)或 `DO NOTHING`(跳过)。

**`EXCLUDED`** = 本次想插、但因冲突被"排除"掉的那行 → 即**新值(the incoming row)**。对照:
| 写法 | 指 | 例子里 |
|---|---|---|
| `EXCLUDED.height_cm` | 新值(想插的) | 178 |
| `height_cm` / `patients.height_cm` | 旧值(表里的) | 现存身高 |

能两个一起用,如累加:`SET n = patients.n + EXCLUDED.n`。

**前提(硬性)**:`ON CONFLICT (列)` 的列**必须有唯一约束/主键**,否则数据库无从判断"撞没撞"。
> 项目实例:[../../src/db/postgres/models.py](../../src/db/postgres/models.py) `upsert_raw_document` = `ON CONFLICT (source_id) DO UPDATE`,实现**幂等写入(idempotent write)**(同书重灌不报错不重复)。

---

## 8. 吞掉冲突的场景(when to swallow the conflict)

"吞掉冲突" = 重复出现时不报错,而是当成正常情况处理。**当"重复"是预期/正常时,就该吞**:

| 场景 | 怎么用 |
|---|---|
| **幂等重试 / idempotent retry**(脚本重跑、接口重发) | `DO NOTHING/UPDATE`,重复执行无害 |
| **并发竞态 / race condition**(两请求同时插) | `ON CONFLICT` 原子处理,见下节 |
| **配置/状态同步 / state sync**("我只要它现在是这个值") | `DO UPDATE SET value = EXCLUDED.value` |
| **去重导入 / dedup import**(已有跳过) | `DO NOTHING` |
| **计数累加 / counter** | `DO UPDATE SET n = 表.n + 1` |

**反过来,什么时候别吞、要报错**:重复是**意料外的错误信号(error signal)**时。如用户注册邮箱已存在 → 要报错告诉用户;纯插入日志/审计(`rag_trace`)撞重复说明上游 bug → 让它报错暴露。

> 判断标准一句话:**重复是"正常预期"就吞(UPSERT),是"错误信号"就别吞(普通 INSERT 让它报错)。**

---

## 9. UPSERT 为什么并发安全(深入 / concurrency safety)

> 🟡 **了解即可(对实习是锦上添花)**:理解"为什么 UPSERT 原子安全"这个**结论**很值钱(面试能讲);但内部**术语名**(speculative insertion / gap lock / CAS / TOCTOU)**不必背**,知道有这回事即可。先学必学,本节可略读。

**问题**:应用层"先查再插"(check-then-act,也叫 TOCTOU)有竞态——

```
请求A: 查(没有) ………… 插入✅
请求B:      查(也没有) ………… 插入💥撞键
                  ↑ 这条缝里 B 也查到"没有"
```
两条独立语句,中间有缝;`SELECT` 只读**不加锁(no lock)**(不存在的行根本锁不住),所以挡不住。

**UPSERT 为什么没缝**:它是**一条语句**,"检查"不是单独的读,而是**"去抢唯一索引那把锁(index lock)"这个动作本身**——抢到=不存在(占住),抢不到=已存在(冲突)。**检查和占用是同一个动作,锁从检查一直握到写完,中间没有"查完还没锁"的瞬间。**

> **拧门把手类比**:先查再插 = 看一眼门牌"空闲"→ 走开 → 回来挂"占用"牌(中间门没锁,别人也能占);UPSERT = 直接拧门把手锁门,拧得动=占成功,拧不动=已有人,"检查能不能锁"和"锁上"是同一个动作。

**最关键的一点**:普通行锁(row lock)只能锁**已存在的行**;唯一索引能在写入时**给一个还不存在的 key 占坑加锁**——这正是行锁做不到、而它能挡住"两个都想新建同一个 key"竞态的原因。

**精确归位**:
- 真正保证"绝不出现两条重复"的,是**唯一约束 + 索引锁(unique constraint + index lock)**(原子的)。哪怕普通 `INSERT`(带唯一约束)也不会插出重复,只是输家**报错**。
- `ON CONFLICT` 只决定**输家怎么收场**:报错 → 改成优雅的 DO NOTHING / DO UPDATE。

术语:这是原子的 **compare-and-set(CAS)**,不是 check-then-act;PostgreSQL 内部叫 **speculative insertion**(试探插入:先占槽位,撞了再退回走 ON CONFLICT)。给"不存在的 key 占坑加锁"防并发插入,概念上对应 MySQL InnoDB 的 **gap lock**(防幻读插入)。

---

## 10. 主键 vs 唯一约束(primary key vs unique constraint)

| | 主键 PK | 唯一约束 unique |
|---|---|---|
| 数量 | 全表**只能一个** | 可以**多个** |
| 空值 | **不允许 NULL** | 通常**允许 NULL**(多个 NULL 不算重复) |
| 用途 | 行的"身份(identity)" | 保证某列不重复 |

- 一张表只有一个主键,但能有多个唯一列。项目实例:[../../src/db/postgres/models_patient.py](../../src/db/postgres/models_patient.py) 的 `users` 表 = 主键 `id` + 唯一列 `email`。
- **UPDATE 不需要主键**(靠 WHERE 任意列找行);**UPSERT 必须有唯一约束**(靠它判冲突)。主键只是唯一约束最常见的一种。
- 主键"必要"的硬场景:**UPSERT、被外键引用(foreign key)**;此外"精确改/删一行、防重复、性能、ORM 映射"也强烈需要,所以实践默认每表配主键。

---

## 11. 易错点速查(common pitfalls)

| 坑 | 正确做法 |
|---|---|
| `WHERE col = NULL` 查不到 | 用 `IS NULL` / `IS NOT NULL`(NULL 不等于任何值) |
| `UPDATE`/`DELETE` 忘 `WHERE` → 全表 | 先用同条件 `SELECT` 验行数,批量改删包事务 |
| 以为分行是语法要求 | 空白不敏感,分行纯为可读;`;` 才是结束 |
| 纯插入也用 UPSERT | 意图要准:纯插用 `INSERT`(撞重复时往往**该**报错暴露 bug) |
| `ON CONFLICT` 的列没唯一约束 | 必须先有唯一约束/主键,否则语句报错 |
| 假设自增 id 连续无空洞 | id 唯一且递增,但回滚/并发会留 gaps,别依赖连续 |

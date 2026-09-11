# PoseSearch：国漫 3D 角色图包独立姿态检索服务

版本：设计稿 v1.0  
核查日期：2026-09-10  
定位：可以交给开发 Agent 实施的技术规格；不是已完成的软件，也不是已在目标 NAS 或用户图库上验证过的性能报告。

## 1. 最终决策

采用以下唯一默认技术路线：

```text
YOLOX-tiny（HumanArt + COCO 人体检测权重，416×416）
  → RTMPose-s（body7，17 个 COCO 身体关键点，输入 H×W=256×192）
  → 可靠性检查与分部位几何特征
  → SQLite 持久化 + NumPy 内存矩阵
  → 带有效性掩码、镜像选项和覆盖率门槛的姿态检索
  → FastAPI HTTP API
```

模型推理使用 ONNX Runtime CPU / FP32。复用 RTMLib 的模型推理和前后处理代码，增加小型适配层。默认部署为一个 Docker 服务，内部一个 API 进程、一个推理 Worker 进程；仅 Worker 持有模型。

第一版不使用 CLIP、DINO、视觉语言大模型、3D 人体重建、手指/脸部全身模型、Redis、Celery、Qdrant、Milvus 或 HNSW。第一版也不把 FAISS 当作必要依赖。

这些取舍是本项目的设计决策，并非声称上述技术没有用途。

### 目标与边界

目标是在尽量忽略脸、服饰与背景的情况下，根据一张图片中指定角色的身体几何姿态，寻找图库中其他角色的相似姿态图片。

默认实现的是“二维画面中的身体姿态相似”，不是视频动作理解，也不是任意视角下的真实三维动作等价。3D 渲染风格的图片仍然只提供一个二维视图。

默认场景是以单人为主的图库，但数据模型必须支持一张图片中的多个人体实例。NAS 目标硬件为 x86-64、i3-8100、整机内存 4–8 GB，不依赖 GPU。第一阶段以十万人体实例规模为设计和压测基线；这不是用户已确认的实际图库数量。

## 2. 为什么采用这个模型组合

RTMLib 的 `Body` 轻量预设已经明确采用 YOLOX-tiny HumanArt 检测器和 RTMPose-s body7 姿态模型，且提供 ONNX Runtime CPU 路径。RTMLib 不要求部署完整的 MMCV、MMPose、MMDetection 训练环境。[S1][S2]

HumanArt 面向自然和人工视觉场景，包括二维、三维虚拟人。这使其人体检测权重值得优先在国漫角色图片上测试，但不能据此保证对用户的 AI 生图准确，也不能把检测器的训练背景等同于姿态模型已经针对国漫做过专门适配。[S3]

### 锁定的默认权重

```text
detector:
  yolox_tiny_8xb8-300e_humanart-6f3252f9.zip
  input_hw: [416, 416]

pose:
  rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip
  input_hw: [256, 192]
  rtmlib_input_wh: [192, 256]
  keypoint_schema: coco17
```

上述文件名称与输入尺寸来自核查时的 RTMLib 源码。[S2] 实施时下载、检查模型输入输出，并对实际使用的 ONNX 文件记录 SHA-256。不要只写“使用 lightweight 模式”而允许升级后静默换模型。

创建 `models.lock.json`，至少记录：来源、归档文件名、归档与 ONNX 校验值、输入/输出名称与形状、关键点顺序、推理依赖版本、前处理版本。模型文件预置于只读 `/models`，正常运行阶段不得自动联网下载。

### 必须增加的适配层

不要直接把 `Body(image)` 封装成生产 API 后结束开发：

1. 使用低层检测和姿态接口，保留每个人体的检测框及关键点对应关系。
2. 无检测框时默认返回 `no_person`。核查时的 RTMPose 实现会在 `bboxes` 为空时改为对整图估计；不能让这一行为静默产生“有效人体”。[S4]
3. 明确传入 ONNX Runtime `SessionOptions`。核查时 RTMLib 的基础会话创建代码没有直接传入自定义 SessionOptions，不能误以为在业务配置中写线程数就自动生效。[S5]
4. 最小化修改范围，优先对固定版本建立带测试的小补丁/适配模块；禁止运行时全局 monkey patch。保留上游授权说明。
5. 按锁定版本的代码保持颜色通道、仿射变换、padding、均值方差、SimCC 解码一致。禁止自行猜测 RGB/BGR 或把 H/W 颠倒。

为缺陷诊断预留人工框选人体的分析入口。只有调用方明确提交人体框时，才允许绕过人体检测器；这不等于默认对未检出的图片硬做整图估计。

## 3. 独立服务与相册应用的职责

```text
相册 Web / MT-Anime
  ├─ 维护图片、角色、作品、图包、权限与缩略图
  ├─ 图片新增/变更/删除后通知 PoseSearch
  └─ 调用相似姿态搜索，按返回的 external_id 展示图片

PoseSearch
  ├─ HTTP API + 检索线程
  ├─ SQLite 持久化任务队列与姿态数据
  ├─ 单个推理 Worker
  ├─ 姿态质量评估与几何检索
  └─ 简单诊断接口、CLI、状态与指标
```

服务不依赖相册内部数据库，不修改原图，不移动图片，不自行识别角色身份。`character_id`、`album_id`、`work_id` 都由相册作为可选元数据提供。

服务使用 `(collection_id, external_id)` 作为外部图片身份。相册应使用稳定图片 ID，而不是路径作为身份。改名仅更新路径；内容没有变化时应复用现有提取结果。

### 图片输入

默认使用只读共享目录：宿主机图库映射到容器 `/media/library:ro`。注册多个逻辑根目录时使用 `root_id`，请求只能提交根目录下的相对路径。

支持两类入口：

- 长期建库：`root_id + relative_path + external_id`。
- 临时查询：multipart 上传图片，分析后返回临时 `analysis_id`，默认不加入图库。

第一版不接收任意远程 URL。避免把项目变成一个能够访问任意内网地址的下载代理。

## 4. 图片分析与质量处理

### 4.1 预处理

检查文件类型和解码尺寸，处理 EXIF 朝向，将姿态输出坐标统一映射回“应用 EXIF 后的图片像素坐标”。响应必须返回对应的 `image_width`、`image_height` 和 `coordinate_space`。

建议初始解码限制为单文件 20 MiB、24 百万像素，均可配置。先支持 JPEG、PNG、静态 WebP；其他格式明确返回不支持，不进入无限重试。对透明图片使用固定背景合成策略并纳入 `preprocess_version`。

分析工作图允许保持宽高比缩小到长边 1280 像素；模型再执行自己的检测 letterbox 和人体裁剪预处理。是否调整该上限由样本测试决定。

计算角度、方向和几何距离时，使用像素等比例坐标。不得直接把 `x/W`、`y/H` 当作等比例几何坐标，否则不同画幅会改变肢体方向。用于存储的坐标可以同时除以 `max(W,H)`；展示时再按坐标约定还原。

### 4.2 人体实例

检测与 NMS 后，默认最多分析面积最大的 3 个人体框。达到上限时返回 `persons_truncated=true`，不得隐瞒截断。此上限是保护 NAS 的初始设置。

每个人体单独提取、单独保存、单独检索。默认主体为通过质量检查的人体中检测框面积最大者；调用方可通过 `person_id` 明确选择其他主体。

不要把多个人体的关键点平均为一个向量。图片最终检索排名可以取其中匹配最好的人体，但响应要携带实际匹配的 `person_id`。

### 4.3 关键点与有效性

保留全部 COCO17 原始关键点及模型原始分数。默认检索使用索引 5–16 的 12 个身体点：左右肩、肘、腕、髋、膝、踝。鼻、眼、耳不进入默认身体相似度。

必须区分模型输出分数与“真实可见性/正确概率”。高分关键点也可能是模型对遮挡处的错误推测，不能把分数解释成校准过的识别正确率。

有效性检查包括：模型分数门槛、是否在有效画布内、骨段是否近乎零长度、明显不合理的坐标关系、人体框与关键点整体位置是否冲突。AI 图片可能确实有异常人体比例，所以解剖规则宜保守，不能用真人比例硬过滤所有风格化角色。

每个点、骨段、角度都存有效性掩码。无效值存储时可以用零作占位，但评分中必须排除，绝不把缺失关键点当作原点参与距离。

### 4.4 状态

人体结果区分 `usable_full`、`usable_upper`、`usable_lower`、`low_quality`。图片分析结果可以是 `no_person`、`no_usable_pose`、`ready`、`partial` 或 `failed`。图片解码失败等属于处理错误；没有合格人体属于正常分析结果，不按异常无限重试。

保留排除原因，例如 `insufficient_joints`、`out_of_frame`、`degenerate_geometry`、`unsupported_format`。`low_quality` 结果可以用于诊断，但默认不进入正常搜索。

提供获取关键点和骨架叠加预览的诊断接口。不能仅凭“输出了 17 个点”认定识别成功。

## 5. 姿态特征规范 geometry-v1

不把姿态检索简化为“一个向量 + 一次余弦相似度”。使用结构化几何特征，以及逐项有效性掩码。

### 5.1 固定内容

每个人体的几何数据包括：

- 12 个身体点的二维坐标：24 个 float32。
- 12 条有向骨段的二维单位向量：24 个 float32。
- 8 个关节夹角的余弦：8 个 float32。

合计 56 个 float32。另存有效性掩码、COCO17 原始点及分数、人体框和质量信息。

12 条有向骨段固定为：

```text
左肩→左肘；左肘→左腕
右肩→右肘；右肘→右腕
左髋→左膝；左膝→左踝
右髋→右膝；右膝→右踝
左肩→左髋；右肩→右髋
左肩→右肩；左髋→右髋
```

8 个角度用以下三点、以中间点为顶点定义：

```text
左肩-左肘-左腕；右肩-右肘-右腕
左肘-左肩-左髋；右肘-右肩-右髋
左肩-左髋-左膝；右肩-右髋-右膝
左髋-左膝-左踝；右髋-右膝-右踝
```

骨段方向更少依赖角色肢体长度；坐标结构保留各部位相对位置；角度补充弯曲情况。其作用是工程设计上的互补，不保证对任何图片都优于单一特征。

### 5.2 检索范围

支持 `full`、`upper`、`lower`、`auto`。

`full` 比较 12 个身体点；`upper` 主要比较肩、肘、腕 6 个点；`lower` 比较髋、膝、踝 6 个点。仅使用所选点集能完整定义的骨段/角度。

`auto` 在查询侧根据质量选择一种范围，并在整个查询中保持不变。全身条件不足而上身可用时可降级为 `upper`，但响应必须报告实际范围和原因。候选图片不能各自选择更有利的范围来混排。

显式请求 `full` 时不静默降级。半身图可以与全身图比较上半身，但不能因为腿未出现而宣称“全身姿态一致”。

### 5.3 几何归一化

检索时对查询和每个候选，使用二者共同有效的同名点计算加权中心及均方根尺度，再平移、等比例缩放。两边使用相同点集和相同权重。

这样避免一张以髋部归一化、另一张因缺髋改用肩部，却仍被当作同一坐标体系比较的问题。默认权重为有效点等权；更复杂的置信度加权必须经过独立验证。

尺度退化或共同有效点不足时拒绝该比较。只允许平移和正的等比例缩放；默认不做自由旋转、仿射变换或透视变换。否则站立、侧躺及身体倾斜差异可能被对齐操作抹掉。

### 5.4 距离

以下权重仅为可复现的初始参数，必须在验证集调参，不能称为已经验证的最优值。

```text
D_shape = 0.50 × D_bone + 0.20 × D_angle + 0.30 × D_position
D_final = min(1, D_shape + 0.20 × (1 - coverage))
similarity = 1 - D_final
```

`D_bone`：共同有效骨段的 `(1 - dot(unit_query, unit_candidate)) / 2` 的均值。

`D_angle`：共同有效角度的 `abs(cos_query - cos_candidate) / 2` 的均值。

`D_position`：共同点分别中心化并用同一权重下的 RMS 尺度归一化后，二维坐标的加权平方距离均值除以 4。对于非退化归一化点集，该量在数学上处于 0–1 区间；浮点误差可裁剪。

某一类特征不可用时只能在满足该模式最低要求后，对剩余类别重新归一化权重；不得因只剩一条简单骨段就给出高匹配分数。

`coverage` 定义为候选与查询共同有效的所需关键点数量 / 查询有效的所需关键点数量。这是查询条件下的覆盖率，不必是对称距离。

先应用覆盖率硬门槛，再评分。初始门槛可设 0.75。全身模式还要求上肢、躯干、下肢均有可比较信息；不能只靠“点数够”而放行一个局部匹配。具体点数及骨段门槛在验收集上确定，并版本化记录。

`similarity` 是项目定义的几何排序分，不是百分比正确率。返回 `distance`、`similarity`、`coverage`、实际范围、有效点数和 `metric_version`，不要把 `0.9` 展示成“90% 概率动作相同”。

### 5.5 镜像

支持 `mirror=equivalent|strict`，默认 equivalent。

镜像等价时，分别计算原姿态和镜像姿态的距离，取较小者。镜像操作必须同时处理：x 轴反射、左右关键点语义交换、分数与有效性掩码交换，然后重新计算骨段和角度。不能只反转 x 坐标。

响应返回 `mirror_applied`。不要通过未经约束的任意点重新排列来“解决”左右识别错误；这会产生假匹配。

镜像等价不等于三维视角不变，也不意味着任意正面和背面姿态都应相同。

## 6. 检索执行方案

### 6.1 第一版：NumPy 分块穷举

先按 collection、质量、部位范围及业务元数据筛选候选，然后在 NumPy 连续数组中分块计算上述距离。默认块大小可以从 4096 个人体实例开始测试。查询计算在线程池中执行，不阻塞 API 事件循环；默认同时执行一个 CPU 搜索任务，其余有界排队。

十万人体实例作为起点时，先优先保证掩码和部分匹配正确，而不是引入近似召回的不确定性。不得使用 Python 循环逐张反序列化 JSON、逐张算距离。

之所以不默认使用 FAISS，是本项目距离依赖“查询与候选共同有效的点集”，包含覆盖率、分部位和镜像逻辑，并非天然等价于一个固定向量空间的普通 L2 或余弦距离。FAISS 标准度量及余弦使用条件参见官方文档。[S6]

搜索按图片合并，保留每图匹配最好的人体。自匹配默认排除。可选排除相同 SHA-256 内容，避免完全相同图片换路径后占满结果。

支持业务过滤 `exclude_character_id`，以及结果多样化 `max_per_character`、`max_per_album`。这些依赖调用方提供的角色/图包 ID，不意味着服务具备角色识别能力。缺失 character_id 的图片各自作为独立项处理，不能全部合并进同一个“未知角色”配额。

### 6.2 扩展条件

仅在目标 NAS 上实测：已入库图片查询延迟超过约定预算，且分块、缓存、过滤等基础优化仍不足时，再引入 FAISS 或 HNSW 作为候选召回。

届时必须按部位和可靠性设计召回特征，对直接与镜像查询分别召回，合并足量候选，再用 geometry-v1 精排。用第一版穷举结果测候选 Recall@K，确认近似召回没有丢失重要的部分姿态匹配。

不能把“以后可扩展”实现成第一版同时安装多个向量数据库。

## 7. 持久化与任务机制

### 7.1 建议数据实体

```text
assets
  collection_id, external_id, root_id, relative_path
  content_sha256, size, mtime_ns, source_revision
  character_id, album_id, work_id, metadata_json
  active_extraction_id, deleted_at

extractions
  extraction_id, content_sha256
  detector_hash, pose_hash, preprocess_version, keypoint_schema
  image_width, image_height, status, diagnostics_json

persons
  person_id, extraction_id, person_index, bbox
  keypoints_blob, raw_scores_blob, masks_blob
  geometry_blob, quality, feature_version

jobs
  job_id, job_type, priority, status, payload_json
  attempts, next_run_at, lease_expires_at, heartbeat_at
  source_revision, result_json, error_code, created_at, updated_at

changes
  sequence, entity_type, entity_id, operation

schema_meta
  db_schema_version, active_model_profile, active_feature_version
```

可以调整物理表结构，但不能丢失版本信息、任务恢复、删除一致性和原始关键点。

原始点必须保留。修改相似度权重、特征算法、镜像逻辑时，应优先重建几何特征，而不是把全库图片重新跑神经网络。

不同模型/前处理/特征版本明确隔离。新版本就绪前旧版本继续服务，查询响应记录版本。不同坐标格式或不同归一化方案的数据不得静默混用。

### 7.2 SQLite

SQLite 是唯一持久化事实来源；内存矩阵是可重建缓存。数据库放在 NAS 本地文件系统的数据目录，不得放在 SMB/NFS 挂载文件上。图库可以通过共享目录读取，但这不等于数据库也应放在网络文件系统。SQLite WAL 对部署主机和文件系统有明确限制。[S7]

使用 WAL、短事务、busy timeout，并安排 checkpoint。启动和 CI 检查实际 Python 运行时链接的 SQLite，而不只检查操作系统命令行版本。核查时官方文档已披露 WAL-reset 问题，并指出 3.51.3 及后续版本、以及指定回补版本已修复；应选用已修复且经项目验证的运行时。[S7]

缓存启动时从数据库构建，之后按 `changes.sequence` 增量更新。每次搜索前补齐已提交变更，返回所用缓存序号。写入姿态结果与 change 记录在同一事务中完成。

删除和更新采用版本检查，已失效的旧任务不得覆盖新版本或复活已删除图片。结果返回前检查资产仍有效且特征版本匹配。

### 7.3 队列

API 只入队，单个 Worker 以短事务领取任务，随后释放数据库事务再进行解码、推理。不能在模型推理期间持有 SQLite 写锁。

任务状态至少包含 `queued → running → succeeded/failed/cancelled`。通过租约和心跳恢复异常中断任务；有最大尝试次数和退避。默认失败重试上限可设 3 次。无人体/无可用姿态属于成功完成分析但不可索引，而不是需要反复重试的网络故障。

支持暂停建库、恢复、队列长度限制及交互查询优先级。临时图片分析与批量建库共用 Worker，不启动第二份模型。高优先级任务只在图片之间抢占，避免持续交互导致批量任务永久饥饿。

第一版不使用仅存在于 API 进程内存中的 background task 充当持久化队列。

## 8. HTTP API 契约

统一前缀 `/v1`，使用 API Key；API Key 绑定可访问 collection。业务错误返回稳定的 `error.code`，JSON 不泄露系统绝对路径和内部堆栈。

### 8.1 建库

`POST /v1/assets:batch-upsert`

```json
{
  "collection_id": "mt-anime",
  "items": [
    {
      "external_id": "img_001",
      "source": {
        "root_id": "library",
        "relative_path": "作品A/角色A/图包01/001.png"
      },
      "source_revision": "rev-12",
      "metadata": {
        "character_id": "character_a",
        "album_id": "album_01",
        "work_id": "work_a"
      }
    }
  ]
}
```

返回 `202`，包含每项的 `job_id` 或 `unchanged`。批次建议上限 100，超过时拒绝而不是一次把所有图片读入内存。重复请求使用 stable ID、内容指纹和版本实现幂等。

`source_revision` 是调用方不透明版本标识，不按字符串大小排序。服务为每次有效更新分配单调递增的内部资产版本，Worker 提交时据此做 compare-and-swap。

### 8.2 已入库图片查询

`POST /v1/search`

```json
{
  "collection_id": "mt-anime",
  "query": {
    "external_id": "img_001",
    "person_id": null
  },
  "scope": "auto",
  "mirror": "equivalent",
  "top_k": 30,
  "filters": {
    "exclude_character_id": "character_a"
  },
  "diversify": {
    "max_per_character": 3,
    "max_per_album": 2
  }
}
```

下面是响应字段示例，数值为示意，不是测试结果：

```json
{
  "query": {
    "external_id": "img_001",
    "person_id": "person_q1",
    "effective_scope": "upper",
    "scope_reason": "lower_body_not_usable"
  },
  "metric_version": "geometry-v1.0",
  "index_revision": 1234,
  "results": [
    {
      "external_id": "img_872",
      "person_id": "person_c1",
      "similarity": 0.87,
      "distance": 0.13,
      "coverage": 1.0,
      "matched_joint_count": 6,
      "mirror_applied": true,
      "character_id": "character_b",
      "album_id": "album_20"
    }
  ]
}
```

这个入口只读取已提取特征，不重新加载或运行神经网络。没有足够姿态信息时返回明确错误，例如 `QUERY_POSE_NOT_USABLE`；没有足够好候选时允许返回少于 top_k 的结果，不强行补足。

### 8.3 临时上传查询

`POST /v1/analyses`：multipart 上传，返回 `202 + job_id`。

`GET /v1/jobs/{job_id}`：成功后返回临时 `analysis_id`、人体实例、范围可用性等。

然后通过 `POST /v1/search`，在 query 中使用 `analysis_id`，其余参数一致。`external_id` 与 `analysis_id` 必须且只能提供一个。

临时分析结果初始 TTL 为 1 小时，可配置。原始上传图分析完成后默认删除；若需要叠加预览，在完成时生成并按相同 TTL 清理。临时分析不自动加入图库。

### 8.4 管理和诊断

| 接口 | 作用 |
|---|---|
| `GET /v1/assets/{external_id}?collection_id=...` | 查询分析状态、人体列表、质量及版本 |
| `DELETE /v1/assets/{external_id}?collection_id=...` | 删除索引引用，不删除原图；幂等 |
| `GET /v1/jobs/{job_id}` | 任务状态、错误码、进度 |
| `POST /v1/indexing/pause` / `resume` | 暂停或恢复批量建库 |
| `GET /v1/persons/{person_id}/keypoints` | 原始关键点和坐标约定 |
| `GET /v1/persons/{person_id}/overlay` | 诊断叠加图；受鉴权和访问权限约束 |
| `GET /v1/stats` | 已索引数量、失败率、队列、资源和耗时统计 |
| `GET /health/live` | 进程存活 |
| `GET /health/ready` | 数据库、模型、缓存和 Worker 是否就绪 |

提供 CLI：扫描指定逻辑根目录、批量导入、重建几何特征、导出评测数据、运行 benchmark。CLI 应复用服务内部核心模块，不重新实现第二套特征算法。

## 9. NAS 资源控制与部署

### 默认布局

```text
/media/library  图库，只读
/models         锁定模型，只读
/data           SQLite、临时上传、日志及服务状态，可写
/config         服务配置，只读
```

以一个镜像启动一个 Compose 服务，内部通过明确的父进程管理一个 API 和一个推理子进程，正确转发信号和回收子进程。API 使用一个 Uvicorn worker。不要通过 `--workers 4` 复制模型或全套索引；FastAPI 官方文档明确解释了多进程通常各自占用内存。[S8]

### 建议初始配置

以下是项目拟定义的配置键，不是 RTMLib 原生支持这些 YAML 参数的声明。

```yaml
service:
  api_workers: 1
  inference_workers: 1
  search_concurrency: 1

runtime:
  backend: onnxruntime
  provider: CPUExecutionProvider
  precision: fp32
  intra_op_threads: 2
  inter_op_threads: 1
  execution_mode: sequential
  allow_spinning: false

pipeline:
  analysis_max_long_edge: 1280
  max_persons_per_image: 3
  query_priority_enabled: true
  default_scope: auto
  default_mirror: equivalent

search:
  backend: numpy_exact_masked
  block_size: 4096
  initial_min_coverage: 0.75
  default_top_k: 30
  max_top_k: 100

storage:
  sqlite_path: /data/posesearch.sqlite
  temporary_analysis_ttl_seconds: 3600
```

ORT 会话需要实际设置 `intra_op_num_threads`，使用顺序执行，并关闭 `session.intra_op.allow_spinning` 和 `session.inter_op.allow_spinning`。官方文档说明 spinning 会消耗额外 CPU/功耗；这对 NAS 后台负载有意义。[S9]

同时限制 OpenCV 和 NumPy 使用的数值库线程，避免“两个模型各开线程 + 搜索又开多线程”的过度竞争。容器 CPU 配额与单张任务之间的节流作为额外控制层。

FP32 是兼容性基线。OpenVINO、INT8 或更小模型只能作为有基准对照的后续优化，不把它们预设成一定更快。第一版不依赖核显透传。

### 内存与速度预算

以下为工程预算和验收目标，不是已测性能：

- 默认单 Worker、十万人体实例时，争取总服务峰值内存控制在约 1–1.5 GiB；必须在真实解码负载和 NAS 上实测，之后才能设置可靠硬上限。
- 已入库图片相似搜索，十万人体实例、单并发时，以 P95 不超过 1 秒作为初始优化目标。
- 图片分析必须分别记录读取/解码、检测、每个人体姿态提取、入库时间；暂不承诺单张分析固定毫秒数。

十万人体实例的 56 维 float32 裸特征数据为 `100000 × 56 × 4 = 22,400,000` 字节。另存 COCO17 的 x/y/score 裸数据为 `100000 × 17 × 3 × 4 = 20,400,000` 字节。实际还存在掩码、元数据、数组缓存、临时批次、模型运行时和数据库开销，不能把约 43 MB 裸数据当作服务总内存。

整机只有 4 GB 时，还要给 NAS 系统及其他服务留出空间；判断依据是剩余可用内存和实测峰值，而不是机器标签上的总内存。达不到预算时优先减小并发、解码尺寸、批次和缓存，而不是不停增加模型。

RTMPose 官方报告中的 CPU 基准使用的是其他硬件和测试条件，不能套用为 i3-8100 完整建库吞吐。[S10]

## 10. 安全、恢复与发布要求

对 root_id 做配置白名单，对 relative_path 做规范化和根目录越界检查。拒绝绝对路径、`..` 越界和逃出根目录的符号链接；使用受限只读挂载及非 root 用户进一步限制读取范围。

API Key 在相册后端保存，不直接作为长期密钥嵌入浏览器。默认只暴露到受控内网；需要外部访问时由已有反向代理提供 TLS 与访问控制。

上传有体积、像素数、队列长度和速率上限。临时目录有清理任务和容量限制。错误日志不默认记录图片内容或完整敏感文件路径。

发布锁定 Python 版本、包版本、RTMLib commit、模型 SHA-256、SQLite 实际运行时与容器基础镜像。提供依赖锁文件、数据库迁移、备份恢复方法和模型检查命令。不能在每次容器启动时 `pip install latest`。

备份使用数据库一致性备份方式，不能在 WAL 持续写入时只随意复制主数据库文件。缓存可从数据库恢复，不需要和模型推理结果一起当作不可丢失的事实来源。

项目代码授权、模型权重授权和训练数据授权分别记录。RTMLib 代码采用 Apache-2.0；HumanArt 数据下载页面有非商业用途要求。不能从代码许可证推断全部权重或训练数据可无条件商用或再分发。第一版推理不需要下载训练集；商业发布或打包分发模型前另行核实权重条款。[S3][S11]

## 11. 第一版不承诺的能力

不能承诺从一张静态图可靠判断“正在挥剑攻击”“准备起跳”这类动作意图。静态骨架也不包含武器、交互对象、运动方向和时序信息。

不能承诺细致手势检索，例如手指结印、握拳与张掌；COCO17 主要提供腕部而非手指关节。[S1]

不能承诺明显俯视、仰视、透视缩短、遮挡、非人形角色和多肢体 AI 错误图片中的精确姿态一致。对这些情况应降级、拒绝或明确标记局部匹配。

不能承诺人物正面/侧面/背面的任意视角转换后仍识别为相同真实三维姿态。第一版保留画面方向，以避免把站立与躺卧误当同姿。

可以以后增加“抬手、肘部弯曲”等规则生成的低层姿态标签，但标签必须允许 unknown。不要用一个小 LLM 把不确定关键点包装成听起来很确定的动作结论。

## 12. 验证方案：先证明有效，再扩大开发

### 12.1 模型与数据验证

先从用户真实图库抽样约 400 张。建议按主类别分组：150 张常规全身、100 张半身、100 张困难姿态/遮挡、50 张多人或小主体；每组覆盖不同角色、服饰、背景和生成风格。比例可调整，但不要只取最清晰的正面站姿。

默认模型必须产生骨架叠加图。人工标注主体是否找对、关键肢体是否连对、可用范围、错误原因。分别统计检测失败和姿态失败，避免把两个环节混成一个“准确率”。

为少量困难图片提供人工正确框，测试“正确框 + RTMPose-s”：若关键点因此显著改善，优先改检测阶段；若仍错误，优先处理姿态模型或质量门槛，不先调检索权重。

MediaPipe Pose Lite 可以只在开发评估环境作轻量对照，不成为生产默认依赖。必要时再对照较大 RTMPose 配置，测效果收益与资源成本，不把两个模型长期同时驻留在 NAS。

### 12.2 检索验证

建立约 50 个查询样例及独立候选图库。正例强调“不同角色、不同服饰、相似姿态”；困难负例强调“同一角色/服饰、明显不同姿态”。剔除完全重复图和近重复图的虚假收益。

把调参集与验收集按图包、必要时按生成批次隔离，避免同一套近重复图片同时用于调参与验收。对准备考核 Precision@10 的查询，确认候选库中至少有 10 张人工认可的相关图片；否则不能用达不到的理论上限错误评价系统。

建议区分：身体完整匹配、局部匹配、仅外观相似。报告 P@10 或分级相关性的 nDCG@10，并同时报告识别失败率、查询拒绝率和局部降级率。不能只统计模型挑选出的最容易样本。

初始产品门槛可设：常规全身图片可用骨架比例至少 80%；有充分正例的跨角色查询平均 P@10 至少 0.7。它们是本项目建议验收标准，不是现有模型在用户图库上的已知结果。困难样本单独报告，不用平均值掩盖。

### 12.3 算法单元测试

必须自动验证以下性质：平移不变、等比例缩放不变、不同画幅不改变骨段方向；镜像交换关键点/掩码一致；默认旋转 90 度不会仍当同姿；缺失点不作为零坐标参与；半身不能冒充全身；共同点少时拒绝；退化尺度、零长骨段、NaN/Inf 可控；多人体按实例检索再按图片合并。

### 12.4 系统验收

在真实 NAS 上测试：一万/十万人体实例的冷启动、热查询 P50/P95、分析时查询延迟、推理全流程耗时、峰值 RSS、空闲 CPU、连续批量建库、异常断电/重启恢复、删除和重命名一致性、磁盘空间不足处理、离线启动以及路径越界防护。

没有完成这一步，不能宣称“i3-8100 每张固定 50–200ms”“4 GB 一定足够”或“全库搜索必然毫秒级”。

## 13. 开发顺序与交付物

### 阶段 A：离线正确性

完成模型锁定、检测/姿态适配器、解码坐标约定、骨架可视化、质量状态、geometry-v1 和单元测试。输出真实样本评测记录。模型输出明显错误时不得先投入大量 API/UI 开发。

### 阶段 B：最小完整服务

完成 SQLite 表与迁移、持久化 Worker、批量 upsert、临时分析、相似查询、删除、任务状态、鉴权及 Docker 部署。主应用仅做新增/变更/删除通知和“相似姿态”按钮接入。

### 阶段 C：产品稳定性

完成增量缓存、多样化与跨角色过滤、暂停恢复、运行指标、备份恢复、CLI、NAS 压测及接口文档。

第一版交付物至少包含：

```text
README.md
ARCHITECTURE.md
API.md / 导出的 OpenAPI schema
models.lock.json
依赖锁文件与 LICENSES 清单
Dockerfile + compose.yaml + 配置示例
服务源码、CLI、数据库迁移
几何算法单元测试 + API 集成测试
真实图库评测记录（不混入未经标记的模拟结果）
NAS benchmark 报告
```

只有阶段 C 验收暴露明确需求后，才实施 ANN、手部关键点、外观辅助特征或训练微调。不要提前把它扩成多模型通用图片理解平台。

## 14. 给开发 Agent 的执行约束摘要

构建一个独立的 PoseSearch 项目，按本文默认组合开发。先验证用户真实图片的骨架质量，再完善服务。不要从头训练模型，不要自行改成 CLIP 主检索，不要默认安装大型向量数据库，不要将 API 内存任务当作持久化队列。

保持姿态、质量、检索和 API 各层可独立测试。所有输入尺寸、坐标空间、左右点顺序、掩码语义和模型/特征版本都必须明确。任何性能数据只记录实际测量；目标值标为目标，不填造实测结果。

核心交付标准不是“HTTP 能返回相似度”，而是：相似姿态能够跨角色检索，错误骨架不会静默污染结果，半身/镜像行为可解释，NAS 上能持续稳定运行，而且相册应用只需通过 API 接入。

---

## 参考资料与核查位置

以下均为项目官方仓库、官方文档或作者论文。引用日期为 2026-09-10；后续实施仍需锁定实际验证版本。

[S1] RTMLib 官方仓库：依赖、CPU 后端、关键点模型与模型列表。
```text
https://github.com/Tau-J/rtmlib
```

[S2] RTMLib `Body` 源码：lightweight 检测器、姿态模型和输入大小。
```text
https://raw.githubusercontent.com/Tau-J/rtmlib/main/rtmlib/tools/solution/body.py
```

[S3] HumanArt 官方仓库：人工场景定位、虚拟人类别及数据用途条件。
```text
https://github.com/IDEA-Research/HumanArt
```

[S4] RTMLib RTMPose 源码：空检测框时整图 fallback、裁剪与坐标处理。
```text
https://raw.githubusercontent.com/Tau-J/rtmlib/main/rtmlib/tools/pose_estimation/rtmpose.py
```

[S5] RTMLib BaseTool 源码：ONNX Runtime 会话创建。
```text
https://raw.githubusercontent.com/Tau-J/rtmlib/main/rtmlib/tools/base.py
```

[S6] FAISS 官方度量文档：L2、内积与余弦条件。
```text
https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
```

[S7] SQLite 官方 WAL 文档：本地文件系统、并发约束及 WAL-reset 修复说明。
```text
https://www.sqlite.org/wal.html
```

[S8] FastAPI 官方部署概念：多进程与内存。
```text
https://fastapi.tiangolo.com/deployment/concepts/
```

[S9] ONNX Runtime 官方线程管理：线程数、执行模式与 spinning。
```text
https://onnxruntime.ai/docs/performance/tune-performance/threading.html
```

[S10] RTMPose 作者论文：公开硬件和条件下的评测，不能替代目标 NAS 实测。
```text
https://arxiv.org/abs/2303.07399
```

[S11] RTMLib 代码许可证。
```text
https://raw.githubusercontent.com/Tau-J/rtmlib/main/LICENSE
```

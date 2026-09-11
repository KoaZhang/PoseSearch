# PoseSearch

独立的 **二维身体姿态相似检索服务**，面向国漫/3D AI 角色图库。项目按 `docs/implementation-spec-v1.0.md` 实施，默认技术路线固定为：

`YOLOX-tiny HumanArt → RTMPose-s body7 → geometry-v1 → SQLite + NumPy exact masked search → FastAPI`

第一版不使用 CLIP/DINO、FAISS、HNSW、Qdrant、Milvus、Redis 或 Celery，也不把服饰/人脸外观作为主检索特征。

## 当前开发状态（0.2.0）

已经实现：

- geometry-v1：12 个身体点、12 条有向骨段、8 个关节角度及逐项有效性掩码。
- `full / upper / lower / auto` 范围与覆盖率门槛。
- `mirror=equivalent|strict`，镜像包含 x 反射和左右语义交换。
- NumPy 分块精确检索；共同有效点决定归一化，缺失点不会当作 `(0,0)` 参与评分。
- SQLite WAL 数据层、持久任务队列、租约/心跳字段、资产版本 CAS，旧任务不能覆盖新资产版本。
- FastAPI：批量 upsert、临时图片分析、`analysis_id` 搜索、删除、任务状态、统计、健康检查。
- API 与推理 Worker 分进程；批量建库和临时图片共用同一个推理 Worker/同一份模型。
- `GET /v1/persons/{person_id}/keypoints` 与 `/overlay` 诊断接口。
- 临时上传默认 TTL 1 小时；分析成功后删除原始上传图，overlay 按 TTL 保留，临时结果不加入图库索引。
- RTMLib 低层 YOLOX/RTMPose 适配：空检测框直接返回 `no_person`，不会触发 RTMPose 整图 fallback。
- PoseSearch 自己创建 ONNX Runtime Session，并显式设置线程和 spinning。
- EXIF 方向、JPEG/PNG/静态 WebP、透明图固定白底、文件/像素限制、只读根目录路径越界防护。
- Docker 单服务父进程管理 API + 单推理 Worker。
- 几何、镜像、缺失点、退化姿态、SQLite 队列、路径安全、临时分析、overlay 等自动测试。

### 已验证的真实模型链路

模型文件已做双层 SHA-256 锁定：压缩包和实际提取的 ONNX 都必须匹配 `models.lock.json`。

- YOLOX-tiny HumanArt ONNX: `ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250`
- RTMPose-s body7 ONNX: `9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d`

GitHub Actions 的 `Real model smoke` 已实际下载锁定模型，并对 RTMLib pinned commit 的 `demo.jpg` 完成 CPU ONNX Runtime 推理和 overlay 生成。该样例得到 3 个已分析人体，3 个均通过 `usable_full` 质量门槛。这只证明真实推理链路可运行，**不等同于你的真实国漫图库准确率验收**。

仍未声称完成/验证：

- 尚未在你的真实图库抽样约 400 张做骨架人工验收，因此没有把任何“准确率”写成实测结果。
- pause/resume、备份 CLI、完整 NAS benchmark 属于后续稳定性里程碑。
- 没有在 i3-8100 NAS 上实测，因此 README 不宣称具体 P95、单图毫秒数或内存上限已经达标。

## 本地开发

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

只运行不涉及模型的 API/SQLite/检索开发时，不需要安装 ONNX Runtime/RTMLib。

推理环境：

```bash
pip install -e '.[inference]'
python scripts/resolve_models.py --models-dir models
posesearch models-verify --models-dir models
```

`resolve_models.py` 是显式安装步骤：服务启动时不会自动联网下载。脚本会优先使用 OpenMMLab 官方地址，失败时使用 RTMLib 官方代码中登记的 Hugging Face mirror，并同时校验 archive 与 ONNX SHA-256。

## 配置

```bash
cp config/config.example.yaml config/config.yaml
```

至少修改：

- `roots.library`：宿主机映射后的只读图库根目录。
- `auth.api_keys`：API Key 与可访问 collection。
- `storage.sqlite_path`：必须位于 NAS 本地文件系统，不要把 SQLite 放到 SMB/NFS。

## Docker

```bash
python scripts/resolve_models.py --models-dir models
cp config/config.example.yaml config/config.yaml
# 修改 compose.yaml 的图库挂载路径
docker compose up -d --build
```

默认容器只监听宿主机 `127.0.0.1:8080`；外部访问请通过现有反向代理/TLS/访问控制。

## 两种查询流程

### 已入库图片

1. 相册注册图片：`POST /v1/assets:batch-upsert`。
2. API 创建持久 job；Worker 从只读共享图库读取、推理并保存 extraction/person。
3. 相册轮询 `GET /v1/jobs/{job_id}`。
4. 用户点击“相似姿态”：`POST /v1/search` 使用 `external_id`，只读取已提取特征，不重新跑神经网络。

### 临时图片

1. multipart 上传：`POST /v1/analyses`，得到 `job_id`。
2. 轮询 `GET /v1/jobs/{job_id}`，成功后取得 `analysis_id` 和人体列表。
3. `POST /v1/search` 使用 `analysis_id` 搜图库。
4. 需要人工检查时读取 `/v1/persons/{person_id}/keypoints` 或 `/overlay`。

详见 [API.md](API.md) 与 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 设计原则

- SQLite 是事实来源；内存索引是可重建缓存。
- 原始 COCO17 点和分数保留，几何算法升级时优先重建 geometry，而不是重新跑神经网络。
- `similarity` 是项目定义的排序分，不是“动作相同概率”。
- 半身不冒充全身；显式 `scope=full` 不静默降级。
- 任何性能数字只记录真实 benchmark；目标值不伪装成实测。

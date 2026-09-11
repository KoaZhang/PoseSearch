# Architecture

## Process layout

```text
MT-Anime / album backend
        |
        | HTTP / API key
        v
+-------------------------------+
| PoseSearch container          |
|                               |
|  API process (1 uvicorn)      |
|    - validation/auth          |
|    - SQLite short writes      |
|    - NumPy search cache       |
|                               |
|  Inference worker (1)         |
|    - YOLOX-tiny HumanArt      |
|    - RTMPose-s body7          |
|    - quality + geometry-v1    |
+-------------------------------+
   |         |             |
 /data    /models      /media/library
 local    read-only       read-only
 SQLite   locked ONNX     image library
```

The API process never loads neural-network models. The worker is the single model owner. The service supervisor starts exactly one of each and forwards termination signals.

## Analysis flow

1. Resolve `root_id + relative_path` under a configured read-only root and reject traversal/symlink escape.
2. Decode JPEG/PNG/static WebP, apply EXIF orientation and optional 1280-long-edge work resize.
3. Human detection with YOLOX-tiny HumanArt.
4. If no boxes: return `no_person`. **Do not call RTMPose with an empty bbox list.**
5. Keep at most the three largest boxes and report truncation.
6. RTMPose-s on each selected bbox.
7. Map keypoints back to EXIF-oriented original-pixel coordinates.
8. Conservative validity/quality checks, then build geometry-v1.
9. Commit extraction/person rows only when `asset_version` still matches the job (CAS).
10. Insert a `changes` row in the same transaction so the API can rebuild/refresh its in-memory cache.

## geometry-v1

Stored per person:

- 12 BODY points = 24 float32.
- 12 directed unit bone vectors = 24 float32.
- 8 joint-angle cosines = 8 float32.
- Validity masks for point/bone/angle features.
- Raw COCO17 keypoints + raw model scores for diagnosis/rebuilds.

Search does **candidate-dependent common-point normalization**. Query and candidate each use their own center/RMS scale, but both are computed from the exact same set of jointly valid named points. No free rotation, affine or perspective alignment is performed.

## Exact search

The cache stacks geometry into contiguous NumPy arrays and scores candidates in blocks. The metric is:

```text
D_shape = 0.50*D_bone + 0.20*D_angle + 0.30*D_position
D_final = min(1, D_shape + 0.20*(1-coverage))
similarity = 1-D_final
```

Unavailable feature classes are reweighted only after minimum scope requirements pass. `full` additionally requires comparable upper, torso and lower information. Mirror equivalence scores direct and mirrored query geometry and chooses the lower distance.

## Persistence and recovery

- SQLite WAL, local filesystem only.
- API transactions are short; inference never holds a SQLite write lock.
- Jobs use `queued/running/succeeded/failed`, attempts, lease and heartbeat fields.
- Stale asset jobs cannot overwrite a newer asset because extraction commit checks the internal monotonic `asset_version`.
- In-memory search arrays are rebuildable from SQLite.

## Deliberately deferred

Temporary upload search, overlay images, pause/resume, backup/restore commands, full change-sequence incremental patching (current 0.1 rebuilds the cache when revision changes), and ANN candidate recall are later milestones.

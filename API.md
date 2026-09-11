# API (0.1.0)

Prefix: `/v1`. Send `X-API-Key` for collection-scoped operations.

## Batch upsert

`POST /v1/assets:batch-upsert`

```json
{
  "collection_id": "mt-anime",
  "items": [
    {
      "external_id": "img_001",
      "source": {"root_id": "library", "relative_path": "作品A/角色A/001.png"},
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

Returns `202`. Changed items receive `job_id`; identical registrations return `unchanged: true`.

## Search indexed asset

`POST /v1/search`

```json
{
  "collection_id": "mt-anime",
  "query": {"external_id": "img_001", "person_id": null},
  "scope": "auto",
  "mirror": "equivalent",
  "top_k": 30,
  "filters": {"exclude_character_id": "character_a"},
  "diversify": {"max_per_character": 3, "max_per_album": 2}
}
```

Returns `metric_version`, `index_revision`, effective scope/reason, and results containing `distance`, `similarity`, `coverage`, matched joints and `mirror_applied`.

`similarity` is a geometry ranking score, not a calibrated probability.

## Other implemented endpoints

- `GET /v1/assets/{external_id}?collection_id=...`
- `DELETE /v1/assets/{external_id}?collection_id=...`
- `GET /v1/jobs/{job_id}`
- `GET /v1/stats`
- `GET /health/live`
- `GET /health/ready`

## Planned but not implemented in 0.1.0

- `POST /v1/analyses` multipart temporary upload.
- `analysis_id` search (currently returns `501 NOT_IMPLEMENTED`).
- `GET /v1/persons/{person_id}/keypoints` and `/overlay`.
- `POST /v1/indexing/pause` and `/resume`.

FastAPI exports the live OpenAPI schema at `/openapi.json` once the service runs.

# API (0.2.0)

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

## Temporary image analysis

`POST /v1/analyses` uses multipart form data:

- `collection_id`: target collection used for authorization/search scope.
- `image`: JPEG, PNG, or static WebP.
- `include_overlay`: optional boolean, default `true`.

Returns `202`:

```json
{"job_id":"..."}
```

Poll `GET /v1/jobs/{job_id}`. A successful result contains `analysis_id`, `status`, `persons`, and whether an overlay was produced. The upload is analyzed by the same persistent inference worker used for library indexing; the raw upload is deleted after successful processing. Temporary analyses expire according to `temporary_analysis_ttl_seconds` and are never inserted into the library search index.

`GET /v1/analyses/{analysis_id}?collection_id=...` returns the current temporary-analysis record while it is live.

## Search

`POST /v1/search`

Indexed-asset query:

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

Temporary-analysis query uses the same request with:

```json
"query": {"analysis_id": "...", "person_id": null}
```

`external_id` and `analysis_id` are mutually exclusive. Results contain `distance`, `similarity`, `coverage`, matched joints and `mirror_applied`. `similarity` is a geometry ranking score, not a calibrated probability.

## Diagnostics

- `GET /v1/persons/{person_id}/keypoints` returns COCO17 x/y coordinates, raw scores, valid mask, quality, scopes, image dimensions and coordinate-space convention.
- `GET /v1/persons/{person_id}/overlay` returns a JPEG diagnostic overlay. For temporary analyses it serves the TTL-scoped generated overlay; for indexed assets it renders from the authorized source image on demand.

## Other endpoints

- `GET /v1/assets/{external_id}?collection_id=...`
- `DELETE /v1/assets/{external_id}?collection_id=...`
- `GET /v1/jobs/{job_id}`
- `GET /v1/stats`
- `GET /health/live`
- `GET /health/ready`

## Not yet implemented

- `POST /v1/indexing/pause` and `/resume`.
- Full NAS benchmark/acceptance report on the target hardware.

FastAPI exports the live OpenAPI schema at `/openapi.json` once the service runs.

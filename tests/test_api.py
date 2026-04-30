import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_list_documents_empty(client: AsyncClient):
    response = await client.get("/api/v1/documents/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_ingest_invalid_file_type(client: AsyncClient):
    response = await client.post(
        "/api/v1/documents/ingest",
        files={"file": ("test.xyz", b"content", "application/octet-stream")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_search_empty_index(client: AsyncClient):
    response = await client.post(
        "/api/v1/search/",
        json={"query": "what is diabetes", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["results"] == []

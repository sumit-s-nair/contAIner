import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from src.mcp.models import DocRequest, DocChunk
from src.mcp.adapters.pip_adapter import PipAdapter
from src.mcp.adapters.npm_adapter import NpmAdapter
from src.mcp.adapters.maven_adapter import MavenAdapter
from src.mcp.adapters.go_adapter import GoAdapter
from src.mcp.adapters.docker_adapter import DockerAdapter
from src.mcp.adapters.cargo_adapter import CargoAdapter
from src.mcp.adapters.conda_adapter import CondaAdapter
from src.mcp.adapters.brew_adapter import BrewAdapter
from src.mcp.adapters.apt_adapter import AptAdapter
from src.mcp.adapters.base import BaseAdapter

# ── Mock HTML responses ──────────────────────────────────────────

MOCK_HTML = "<html><body><h2>Usage</h2><pre>command syntax</pre></body></html>"

@pytest.fixture
def mock_http():
    """Mock the HTTP layer in BaseAdapter to prevent real network calls."""
    with patch.object(BaseAdapter, '_fetch_html', new_callable=AsyncMock) as mock_html:
        with patch.object(BaseAdapter, '_fetch_json', new_callable=AsyncMock) as mock_json:
            mock_html.return_value = MOCK_HTML
            mock_json.return_value = {}
            yield mock_html, mock_json


# ── Smoke tests for all 9 adapters ───────────────────────────────

@pytest.mark.asyncio
async def test_pip_adapter(mock_http):
    mock_html, mock_json = mock_http
    adapter = PipAdapter()
    req = DocRequest(tool="pip", operation="install", package="requests")
    chunk = await adapter.fetch(req)
    
    assert isinstance(chunk, DocChunk)
    mock_html.assert_called_once()
    mock_json.assert_not_called()

@pytest.mark.asyncio
async def test_npm_adapter(mock_http):
    mock_html, mock_json = mock_http
    adapter = NpmAdapter()
    req = DocRequest(tool="npm", operation="install", package="react")
    chunk = await adapter.fetch(req)
    
    assert isinstance(chunk, DocChunk)
    mock_html.assert_called_once()
    mock_json.assert_not_called()

@pytest.mark.asyncio
async def test_go_adapter(mock_http):
    mock_html, mock_json = mock_http
    adapter = GoAdapter()
    req = DocRequest(tool="go", operation="get", package="github.com/gin-gonic/gin")
    chunk = await adapter.fetch(req)
    
    assert isinstance(chunk, DocChunk)
    mock_html.assert_called_once()
    mock_json.assert_not_called()

@pytest.mark.asyncio
async def test_docker_adapter(mock_http):
    mock_html, mock_json = mock_http
    adapter = DockerAdapter()
    req = DocRequest(tool="docker", operation="run", package="ubuntu")
    chunk = await adapter.fetch(req)
    
    assert isinstance(chunk, DocChunk)
    mock_html.assert_called_once()
    mock_json.assert_not_called()

@pytest.mark.asyncio
async def test_cargo_adapter(mock_http):
    mock_html, mock_json = mock_http
    adapter = CargoAdapter()
    req = DocRequest(tool="cargo", operation="add", package="serde")
    chunk = await adapter.fetch(req)
    
    assert isinstance(chunk, DocChunk)
    mock_html.assert_called_once()
    mock_json.assert_not_called()

@pytest.mark.asyncio
async def test_conda_adapter(mock_http):
    mock_html, mock_json = mock_http
    adapter = CondaAdapter()
    req = DocRequest(tool="conda", operation="install", package="numpy")
    chunk = await adapter.fetch(req)
    
    assert isinstance(chunk, DocChunk)
    mock_html.assert_called_once()
    mock_json.assert_not_called()

@pytest.mark.asyncio
async def test_brew_adapter(mock_http):
    mock_html, mock_json = mock_http
    adapter = BrewAdapter()
    req = DocRequest(tool="brew", operation="install", package="wget")
    chunk = await adapter.fetch(req)
    
    assert isinstance(chunk, DocChunk)
    mock_html.assert_called_once()
    mock_json.assert_not_called()

@pytest.mark.asyncio
async def test_apt_adapter(mock_http):
    mock_html, mock_json = mock_http
    adapter = AptAdapter()
    req = DocRequest(tool="apt", operation="install", package="curl")
    chunk = await adapter.fetch(req)
    
    assert isinstance(chunk, DocChunk)
    mock_html.assert_called_once()
    mock_json.assert_not_called()

@pytest.mark.asyncio
async def test_maven_adapter(mock_http):
    mock_html, mock_json = mock_http
    adapter = MavenAdapter()
    req = DocRequest(tool="maven", operation="add", package="com.google.guava:guava")
    chunk = await adapter.fetch(req)
    
    assert isinstance(chunk, DocChunk)
    mock_html.assert_called_once()
    mock_json.assert_not_called()


# ── Specific Maven string parsing tests ─────────────────────────

@pytest.mark.asyncio
async def test_maven_adapter_parsing_with_colon(mock_http):
    """Test given com.google.guava:guava group_id/artifact_id split correctly."""
    adapter = MavenAdapter()
    req = DocRequest(tool="maven", operation="add", package="com.google.guava:guava")
    chunk = await adapter.fetch(req)
    
    # Check that the command_syntax contains the separated parts correctly
    assert "<groupId>com.google.guava</groupId>" in chunk.command_syntax
    assert "<artifactId>guava</artifactId>" in chunk.command_syntax

@pytest.mark.asyncio
async def test_maven_adapter_parsing_without_colon(mock_http):
    """Test bare artifact name with no colon uses placeholder for group_id."""
    adapter = MavenAdapter()
    req = DocRequest(tool="maven", operation="add", package="guava")
    chunk = await adapter.fetch(req)
    
    # Check that the command_syntax contains placeholder for groupId and guava for artifactId
    assert "<groupId>{groupId}</groupId>" in chunk.command_syntax
    assert "<artifactId>guava</artifactId>" in chunk.command_syntax


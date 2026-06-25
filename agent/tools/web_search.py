import re
from typing import Any

import httpx

from agent.tools.base import BaseTool, ToolResult
class WebSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web via DuckDuckGo. Returns up to 5 results with title, snippet, and URL."
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type" : "object",
            "properties":{
                "query": {"type": "string", "description": "Search query",},
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        if not query:
            return ToolResult(success=False, output=None, error="No query provided")
        
        try:
            url = "https://html.duckduckgo.com/html/"
            response = httpx.post(
                url,
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            response.raise_for_status()

            results = self._parse_results(response.text)
            return ToolResult(
                success=True,
                output=results[:5],
                metadata={"total_found": len(results)},
            )
        except httpx.HTTPError as e:
            return ToolResult(success=False, output=None, error=f"HTTP error: {e}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    def _parse_results(self, html: str) -> list[dict[str, str]]:
        results = []
        blocks = re.findall(r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</(?:a|span|div)', html, re.DOTALL)
        for url, title, snippet in blocks:
            title = re.sub(r"<[^>]+>", "", title).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            if url.startswith("//"):
                url = "https:" + url
            results.append({"title": title, "snippet": snippet, "url": url})
        return results
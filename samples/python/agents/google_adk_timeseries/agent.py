import requests
from typing import Any, Optional
from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.function_tool import FunctionTool

from task_manager import AgentWithTaskManager


class StockTimeSeriesAgent(AgentWithTaskManager):
    """An agent that performs time series analysis on stock data."""


    SUPPORTED_CONTENT_TYPES = ["text/plain", "text/markdown"]  # 加上這一行
    def __init__(self):
        self.api_key = "PQ473T4WH99XBU3Z"  # 你需要填入自己的 API KEY
        self._agent = self._build_agent()

        self._user_id = 'stock_timeseries_agent'
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
    def get_processing_message(self) -> str:
        return 'Processing the stock timeseries request...'

    
    def fetch_historical_data(self, symbol: str, start_date: str = None, end_date: str = None, outputsize: str = "compact") -> dict:
        """Fetches historical daily data from Alpha Vantage API and filters by date range."""
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            return {"error": data.get("Note") or "API 回應異常"}

        time_series = data["Time Series (Daily)"]

        # 若有指定日期範圍，做篩選
        if start_date and end_date:
            filtered = {date: info for date, info in time_series.items() if start_date <= date <= end_date}
            if not filtered:
                return {"error": "查無指定日期區間資料"}
            return filtered
        
        # 若沒指定，直接回傳全部（compact 或 full）資料
        return time_series
    
    

    def analyze_trend(self, symbol: str, start_date: str, end_date: str) -> dict:
        """
        Performs a basic trend and volatility analysis on the specified date range.
        Dates must be in 'YYYY-MM-DD' format.
        """
        data = self.fetch_historical_data(symbol, start_date=start_date, end_date=end_date)
        if "error" in data:
            return {"status": "error", "message": data["error"]}
        
        # 按日期由舊到新排序（Alpha Vantage 回傳是 dict，key 是日期字串）
        try:
            sorted_dates = sorted(data.keys())
            closes = [float(data[date]["4. close"]) for date in sorted_dates]
            if len(closes) < 2:
                return {"status": "error", "message": "指定日期範圍內資料不足"}
        except Exception as e:
            return {"status": "error", "message": f"資料解析錯誤：{e}"}
        
        trend = "up" if closes[-1] > closes[0] else "down"
        avg_change = round(sum([(closes[i+1] - closes[i]) for i in range(len(closes)-1)]) / (len(closes)-1), 3)
        volatility = round(max(closes) - min(closes), 2)
        
        return {
            "status": "success",
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "trend": trend,
            "avg_daily_change": avg_change,
            "volatility": volatility,
        }

    

    def _build_agent(self) -> LlmAgent:
        instruction = """
        You are a stock analysis agent specializing in time series data.
        When a user provides a stock symbol, use your tools to:
        - Fetch historical price data
        - Analyze recent trends (up/down)
        - Calculate volatility and average daily price change
        Provide clear analysis based on the data.
        Do not say you cannot access external data — use the `analyze_trend` tool to perform the analysis.
        """
        return LlmAgent(
            name="stock_time_series_agent",
            model="gemini-2.5-flash",
            description="Agent that uses time series tools to analyze stock data",
            instruction=instruction,
            tools=[self.analyze_trend],
        )

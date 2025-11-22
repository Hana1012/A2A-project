import requests
from datetime import datetime
from typing import Any
from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from task_manager import AgentWithTaskManager

class MarketDataAgent(AgentWithTaskManager):
    """An agent that fetches real-time market data."""

    SUPPORTED_CONTENT_TYPES = ["text/plain", "text/markdown"]

    def __init__(self):
        self.api_key = "PQ473T4WH99XBU3Z"  
        self._agent = self._build_agent()
        self._user_id = 'market_data_agent'
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def get_processing_message(self) -> str:
        return 'Fetching stock price...'

    def fetch_stock_data(self, stock_symbol: str, start_date: str, end_date: str) -> dict[str, Any]:
        """Fetches historical stock data from Alpha Vantage and filters by date range."""
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": stock_symbol,
            "outputsize": "full",
            "apikey": self.api_key,
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # 處理 API 回傳限制或錯誤
            if "Note" in data:
                return {"status": "error", "message": data["Note"]}
            if "Error Message" in data:
                return {"status": "error", "message": data["Error Message"]}
            if "Time Series (Daily)" not in data:
                return {"status": "error", "message": "API 回傳格式無效或沒有找到資料。"}

            historical_data = data["Time Series (Daily)"]

            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            filtered_data = []
            for date_str, daily_info in historical_data.items():
                current_dt = datetime.strptime(date_str, '%Y-%m-%d')
                if start_dt <= current_dt <= end_dt:
                    filtered_data.append({
                        "date": date_str,
                        "open": float(daily_info["1. open"]),
                        "high": float(daily_info["2. high"]),
                        "low": float(daily_info["3. low"]),
                        "close": float(daily_info["4. close"]),
                        "volume": float(daily_info["5. volume"])
                        
                    })

            if not filtered_data:
                return {"status": "error", "message": f"在 {start_date} 到 {end_date} 期間沒有找到資料。"}

            # 依日期排序
            filtered_data.sort(key=lambda x: x["date"])

            return {
                "status": "success",
                "stock_symbol": stock_symbol,
                "data": filtered_data
            }

        except (requests.exceptions.RequestException, ValueError) as e:
            return {"status": "error", "message": f"無法獲取或處理資料: {e}"}

    def _build_agent(self) -> LlmAgent:
        instruction = (
            "You are a specialized agent for fetching market data. "
            "Use the provided tool to fetch the latest stock price "
            "and return the result as structured data."
        )

        return LlmAgent(
            model='gemini-2.5-flash',
            name='market_data_agent',
            description='Provides real-time stock market data.',
            instruction=instruction,
            tools=[self.fetch_stock_data],
        )

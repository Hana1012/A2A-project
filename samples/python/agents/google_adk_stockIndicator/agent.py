import requests
import numpy as np
#import pandas as pd
from typing import Any, List, Optional
from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.function_tool import FunctionTool
from agents.google_adk_market.agent import MarketDataAgent

from task_manager import AgentWithTaskManager
from datetime import datetime, timedelta

class StockIndicatorAgent(AgentWithTaskManager):
    """An agent that performs stock investment indicator analysis."""

    SUPPORTED_CONTENT_TYPES = ["text/plain", "text/markdown"]

    def __init__(self):
        self.api_key = "PQ473T4WH99XBU3Z" 
        self._agent = self._build_agent()
        self._user_id = 'analyze_stock_indicators'
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def get_processing_message(self) -> str:
        return 'Analyzing stock investment indicators...'

    # ===== 技術指標 =====
    def _sma(self, closes: List[float], window: int = 14) -> float:
        return round(np.mean(closes[-window:]), 2) if len(closes) >= window else None

    def _rsi(self, closes: List[float], window: int = 14) -> float:
        if len(closes) < window + 1:
            return None
        diffs = np.diff(closes[-(window+1):])
        gains = diffs[diffs > 0].sum() / window
        losses = -diffs[diffs < 0].sum() / window
        rs = gains / losses if losses != 0 else np.inf
        return round(100 - (100 / (1 + rs)), 2)

    def _macd(self, closes: List[float], short=12, long=26, signal=9) -> dict[str, float]:
        if len(closes) < long + signal:
            return None
        ema_short = pd.Series(closes).ewm(span=short, adjust=False).mean()
        ema_long = pd.Series(closes).ewm(span=long, adjust=False).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return {
            "macd": round(macd_line.iloc[-1], 3),
            "signal": round(signal_line.iloc[-1], 3),
            "hist": round(hist.iloc[-1], 3)
        }

    def _bollinger_bands(self, closes: List[float], window: int = 20) -> dict[str, float]:
        """計算布林通道 (上軌/中軌/下軌)"""
        if len(closes) < window:
            return None
        sma = np.mean(closes[-window:])
        std = np.std(closes[-window:])
        upper = sma + 2 * std
        lower = sma - 2 * std
        return {
            "middle": round(sma, 2),
            "upper": round(upper, 2),
            "lower": round(lower, 2)
        }

    # ===== 分析整合 =====
    def analyze_indicators(self, symbol: str, start_date: str, end_date: str) -> dict:
        """分析 SMA, RSI, MACD, Bollinger Bands, 波動率。"""
        # 👈 核心修正：使用新的 call_agent 方法來獲取數據
        #data_result = self.call_agent("market_data_agent", "fetch_stock_data", symbol=symbol, start_date=start_date, end_date=end_date)
        
        """Fetches historical stock data from Alpha Vantage and filters by date range."""
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
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
        except Exception as e:
            return {"status": "error", "message": f"資料獲取過程發生錯誤: {str(e)}"}

        
        df = pd.DataFrame(filtered_data)   # 而不是 data
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        closes = df['close'].tolist()

        
        if len(closes) < 20:
            return {"status": "error", "message": "資料不足以計算投資指標"}
        
        sma_14 = self._sma(closes, 14)
        rsi_14 = self._rsi(closes, 14)
        macd_result = self._macd(closes)
        bollinger = self._bollinger_bands(closes, 20)
        volatility = round(max(closes) - min(closes), 2)
        
        return {
            "status": "success",
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "sma_14": sma_14,
            "rsi_14": rsi_14,
            "macd": macd_result,
            "bollinger_bands": bollinger,
            "volatility": volatility,
        }


    def _build_agent(self) -> LlmAgent:
        instruction = """
        You are a stock analysis agent specializing in investment indicators.
        When a user provides a stock symbol and date range, use your tools to:
        - Fetch historical data
        - Calculate indicators (SMA, RSI, MACD, Bollinger Bands, volatility)
        Provide clear interpretation of the indicators:
        - SMA → 趨勢
        - RSI → 超買/超賣
        - MACD → 動能
        - Bollinger Bands → 支撐/壓力位
        - Volatility → 波動範圍
        """
        return LlmAgent(
            name="stock_time_series_agent",
            model="gemini-2.5-flash",
            description="Agent that calculates investment indicators from stock time series",
            instruction=instruction,
            tools=[self.analyze_indicators],
        )

import requests
import numpy as np
from arch import arch_model
import pandas as pd
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
    
    def fetch_news_sentiment(
        self, 
        tickers: str, 
        time_from: str = None, 
        time_to: str = None, 
        topics: str = None, 
        limit: int = 50, 
        sort: str = "LATEST"
    ) -> dict:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": tickers,
            "apikey": self.api_key,  
            "limit": limit,
            "sort": sort
        }
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to
        if topics:
            params["topics"] = topics

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if "Note" in data:
                return {"status": "error", "message": data["Note"]}
            if "Error Message" in data:
                return {"status": "error", "message": data["Error Message"]}
            return {"status": "success", "data": data.get("feed", [])}
        except Exception as e:
            return {"status": "error", "message": str(e)} 
           

    # ===== 分析整合 =====
    def analyze_indicators(self, symbol: str, start_date: str, end_date: str) -> dict:
        """分析 SMA, RSI, MACD, Bollinger Bands, 波動率。"""
        """Fetches historical stock data from Alpha Vantage and filters by date range."""
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": self.api_key,
        }

        # 1. 取得股價資料
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
        # 2. 計算技術指標
        sma_14 = self._sma(closes, 14)
        rsi_14 = self._rsi(closes, 14)
        macd_result = self._macd(closes)
        bollinger = self._bollinger_bands(closes, 20)
        volatility = round(max(closes) - min(closes), 2)

        # 3. GARCH(1,1) 波動率
        # ===== 新增 GARCH(1,1) 波動性預測 =====
        try:
            returns = df['close'].pct_change().dropna() * 100  # 轉成百分比日報酬率
            garch_model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
            garch_result = garch_model.fit(disp='off')
            # 預測下一日波動率
            garch_forecast = garch_result.forecast(horizon=1)
            garch_volatility = round(np.sqrt(garch_forecast.variance.values[-1, 0]), 4)
        except Exception as e:
            garch_volatility = None
        
        sentiment_all = {}
        for dt in df.index:
            date_str = dt.strftime("%Y%m%dT0000")
            sentiment_res = self.fetch_news_sentiment(
                tickers=symbol,
                time_from=date_str,
                time_to=dt.strftime("%Y%m%dT2359"),
                limit=10
            )
            sentiment_all[dt.strftime("%Y-%m-%d")] = sentiment_res.get("data", [])

        daily_data = {}
        for dt in df.index:
            date_key = dt.strftime("%Y-%m-%d")
            daily_data[date_key] = {
                "close": df.loc[dt, "close"],
                "sma_14": df.loc[dt, "sma_14"],
                "rsi_14": df.loc[dt, "rsi_14"],
                "macd": df.loc[dt, "macd"],
                "macd_signal": df.loc[dt, "macd_signal"],
                "macd_hist": df.loc[dt, "macd_hist"],
                "boll_middle": df.loc[dt, "boll_middle"],
                "boll_upper": df.loc[dt, "boll_upper"],
                "boll_lower": df.loc[dt, "boll_lower"],
                "volatility": df.loc[dt, "volatility"],
                "garch_vol": df.loc[dt, "garch_vol"],
                "news_sentiment": sentiment_all.get(date_key, [])
            }

        return {
            "status": "success",
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "daily_data": daily_data
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

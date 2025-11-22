from datetime import datetime, timedelta
import logging
import requests
from collections import defaultdict, deque
from typing import List, Dict
from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from task_manager import AgentWithTaskManager

logger = logging.getLogger(__name__)

class TradingAgent:
    """Trading agent that executes orders and keeps trade history using MarketAgent prices."""

    SUPPORTED_CONTENT_TYPES = ["text/plain", "text/markdown"]

    def __init__(self, market_agent_url: str):
        self.market_agent_url = market_agent_url
        self.order_book = {"buy": [], "sell": []}  # list of orders
        self.trade_history = []  # executed trades

        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    # ===== Market Price =====
    def get_market_price(self, symbol: str) -> Optional[float]:
        """Query MarketAgent for the latest price of a stock."""
        try:
            resp = requests.get(f"{self.market_agent_url}/price", params={"symbol": symbol})
            data = resp.json()
            if data.get("status") == "success":
                return float(data["price"])
            return None
        except Exception:
            return None

    # ===== Orders =====
    def place_order(self, user_id: str, symbol: str, side: str, quantity: int, price: Optional[float] = None) -> Dict:
        """
        Place a buy/sell order.
        - side: "buy" or "sell"
        - price: if None, use current market price from MarketAgent
        """
        if side not in ["buy", "sell"]:
            return {"status": "error", "message": "Invalid side, must be 'buy' or 'sell'"}

        if price is None:
            price = self.get_market_price(symbol)
            if price is None:
                return {"status": "error", "message": "Cannot fetch market price"}

        order = {
            "order_id": f"{len(self.order_book[side]) + len(self.order_book['buy' if side=='sell' else 'sell']) + 1}",
            "user_id": user_id,
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": quantity,
            "timestamp": datetime.now().isoformat()
        }

        self.order_book[side].append(order)
        self.match_orders(symbol)
        return {"status": "success", "order": order}

    # ===== Order Matching =====
    def match_orders(self, symbol: str):
        """Match buy and sell orders at overlapping prices."""
        buy_orders = sorted([o for o in self.order_book["buy"] if o["symbol"] == symbol], key=lambda x: -x["price"])
        sell_orders = sorted([o for o in self.order_book["sell"] if o["symbol"] == symbol], key=lambda x: x["price"])

        i = 0
        j = 0
        while i < len(buy_orders) and j < len(sell_orders):
            buy = buy_orders[i]
            sell = sell_orders[j]

            if buy["price"] >= sell["price"]:
                # 成交價用賣方價格
                trade_qty = min(buy["quantity"], sell["quantity"])
                trade_price = sell["price"]
                trade = {
                    "symbol": symbol,
                    "price": trade_price,
                    "quantity": trade_qty,
                    "buy_user_id": buy["user_id"],
                    "sell_user_id": sell["user_id"],
                    "timestamp": datetime.now().isoformat()
                }
                self.trade_history.append(trade)

                # 更新剩餘數量
                buy["quantity"] -= trade_qty
                sell["quantity"] -= trade_qty

                if buy["quantity"] == 0:
                    i += 1
                if sell["quantity"] == 0:
                    j += 1
            else:
                break

        # 移除已成交完的訂單
        self.order_book["buy"] = [o for o in buy_orders[i:] if o["quantity"] > 0] + [o for o in self.order_book["buy"] if o["symbol"] != symbol]
        self.order_book["sell"] = [o for o in sell_orders[j:] if o["quantity"] > 0] + [o for o in self.order_book["sell"] if o["symbol"] != symbol]

    # ===== Trade History =====
    def get_trade_history(self, user_id: Optional[str] = None) -> List[Dict]:
        """Return all trades, optionally filtered by user_id."""
        if user_id:
            return [t for t in self.trade_history if t["buy_user_id"] == user_id or t["sell_user_id"] == user_id]
        return self.trade_history

    # ===== Current Order Book =====
    def get_order_book(self, symbol: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Return current order book, optionally filtered by symbol."""
        if symbol:
            return {
                "buy": [o for o in self.order_book["buy"] if o["symbol"] == symbol],
                "sell": [o for o in self.order_book["sell"] if o["symbol"] == symbol]
            }
        return self.order_book
    
    def _build_agent(self) -> LlmAgent:
        instruction =  """
        You are a Simulation Manager agent for a stock market simulation.
        Your role is to control the flow of the simulation across multiple agents.

        Responsibilities:
        1. For each simulation day:
            a. Request the current stock price from the Market Agent.
            b. Request technical indicators from the Stock Indicator Agent.
            c. Send relevant data to the Investment Agent to get a BUY / SELL / HOLD decision.
            d. Send the decision to the Trading Agent to execute the trade and return the result.
        2. Record daily outcomes: price, decision, trade result, and portfolio value.
        3. Repeat until the simulation period ends or termination conditions are met.
        4. Output structured logs suitable for analysis after the simulation.

        Notes:
        - You do not make investment decisions yourself.
        - Focus on orchestrating the agents and maintaining consistent records.
        """
        return LlmAgent(
            name="stock_simulation_agent",
            model="gemini-2.5-flash",
            description="Agent that simulates stock market.",
            instruction=instruction,
            tools=[self.analyze_trend],
        )


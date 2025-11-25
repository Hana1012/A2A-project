import json
import logging
from typing import Any, Optional, List, Dict
from datetime import datetime, timedelta

from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from task_manager import AgentWithTaskManager

from agents.google_adk_market.agent import MarketDataAgent
from agents.google_adk_stockIndicator.agent import StockIndicatorAgent


class InvestmentAgent(AgentWithTaskManager):
    """An agent that simulates a daily multi-stock investment portfolio based on personality."""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self, personality: Optional[dict[str, Any]] = None):
        self.personality = {
            "openness": "medium",
            "conscientiousness": "medium",
            "extraversion": "medium",
            "agreeableness": "medium",
            "neuroticism": "high",
        }

        self._agent = self._build_llm_agent()
        self._user_id = 'investment_agent'
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

        # 註冊其他代理
        self.agents = {
            "google_adk_market": MarketDataAgent(),
            "google_adk_stockIndicator": StockIndicatorAgent(),
        }

    def get_processing_message(self) -> str:
        return 'Simulating daily investment portfolio based on personality...'

    def _build_llm_agent(self) -> LlmAgent:
        instruction = (
            "You are an investment advisor with personality traits:\n"
            f"- Openness: {self.personality['openness']}\n"
            f"- Conscientiousness: {self.personality['conscientiousness']}\n"
            f"- Extraversion: {self.personality['extraversion']}\n"
            f"- Agreeableness: {self.personality['agreeableness']}\n"
            f"- Neuroticism: {self.personality['neuroticism']}\n\n"
            "Your task:\n"
            "1. Each day, decide a multi-stock portfolio using only available cash and current holdings.\n"
            "2. Choose which stocks to BUY, SELL, or HOLD and how many shares for each.\n"
            "3. You do NOT need a predefined stock list; you can pick suitable stocks.\n"
            "4. Output strict JSON with updated portfolio, remaining cash, and natural language explanation."
        )

        return LlmAgent(
            model='gemini-2.5-flash',
            name='daily_investment_agent',
            description='Simulates daily multi-stock investment portfolio decisions based on personality.',
            instruction=instruction,
            tools=[
            #FunctionTool(
            #    func=self.simulate_daily_portfolio
            #)
        ]
        )
        '''return LlmAgent(
            model='gemini-2.5-flash',
            name='daily_investment_agent',
            description='Simulates daily multi-stock investment portfolio decisions based on personality.',
            instruction=instruction,
            tools=[self.simulate_daily_portfolio],
        )'''

    async def call_agent(self, agent_name: str, method_name: str, **kwargs: Any) -> Any:
        """通用代理呼叫方法"""
        try:
            target_agent = self.agents.get(agent_name)
            if not target_agent:
                raise ValueError(f"Agent '{agent_name}' not found. Available: {list(self.agents.keys())}")

            method = getattr(target_agent, method_name, None)
            if not method:
                available_methods = [m for m in dir(target_agent) if not m.startswith('_')]
                raise AttributeError(f"Method '{method_name}' not found on agent '{agent_name}'. Available: {available_methods}")

            if callable(method):
                if hasattr(method, '__await__'):
                    result = await method(**kwargs)
                else:
                    result = method
            else:
                result = method

            return result

        except (ValueError, AttributeError, TypeError) as e:
            logging.error(f"Failed to call agent '{agent_name}': {e}")
            return {"error": str(e)}

    async def simulate_daily_portfolio(
        self,
        start_date: str,
        end_date: str,
        initial_cash: float
    ) -> List[Dict[str, Any]]:
        """
        Simulate a continuous multi-stock portfolio day by day.
        Cash and holdings carry over from previous day. No extra funding is added.
        """
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

        portfolio = {}  # 股票持股數量
        cash = initial_cash
        daily_results = []

        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")

            # 取得每日市場數據與技術指標
            market_data = await self.call_agent("google_adk_market", "fetch_stock_data", date=date_str)
            indicators = await self.call_agent("google_adk_stockIndicator", "analyze_indicators", date=date_str)

            llm_input_messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are an investment advisor with High Openness personality.\n"
                        f"Today is {date_str}. Adjust your portfolio based on market data and technical indicators.\n"
                        f"Current portfolio: {portfolio}, cash available: ${cash:.2f}\n"
                        "Decide which stocks to buy, sell, or hold. Do NOT assume additional funds.\n"
                        "Output JSON with updated portfolio, cash_remaining, and reasoning."
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "market_data": market_data,
                        "technical_indicators": indicators,
                        "current_portfolio": portfolio,
                        "cash_available": cash
                    }, indent=2)
                }
            ]

            # 呼叫 LLM
            #decision_text = await self._runner.run(messages=llm_input_messages)
            decision_text = await self._agent.invoke(input=llm_input_messages)

            # 嘗試解析 JSON
            try:
                decision_json = json.loads(decision_text)
                portfolio = decision_json.get("updated_portfolio", portfolio)
                cash = decision_json.get("cash_remaining", cash)
                explanation = decision_json.get("reason", "")
            except json.JSONDecodeError:
                explanation = "Fallback: LLM output invalid, no trades executed."

            daily_results.append({
                "date": date_str,
                "portfolio": portfolio.copy(),
                "cash_remaining": cash,
                "explanation": explanation,
                "market_data": market_data,
                "indicators": indicators
            })

            current_date += timedelta(days=1)

        return daily_results

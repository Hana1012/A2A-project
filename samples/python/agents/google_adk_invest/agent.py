import json
import logging
from typing import Any, Optional
from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from task_manager import AgentWithTaskManager
from datetime import datetime, timedelta

# 假設你已經有了這些專門的代理
# 這些是 A2A 網絡上的其他「參與者」
from agents.google_adk_market.agent import MarketDataAgent
from agents.google_adk_stockIndicator.agent import StockIndicatorAgent
#from agents.sentiment_analysis_agent import SentimentAnalysisAgent

class InvestmentAgent(AgentWithTaskManager):
    """An agent that coordinates with other agents to make investment decisions."""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self, personality: Optional[dict[str, Any]] = None, personality_type: Optional[str] = "big5"):
        
        self.personality = {
                "openness": "medium",
                "conscientiousness": "low",
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

        
        # 步驟 1: 在這裡實例化並儲存其他代理
        # 這是一個簡易的「註冊表」實現
        self.agents = {
            "google_adk_market": MarketDataAgent(),
            "google_adk_stockIndicator": StockIndicatorAgent(),
            # "SentimentAnalysisAgent": SentimentAnalysisAgent(),
        }


    def get_processing_message(self) -> str:
        return 'Analyzing market data and making a personality-driven investment decision...'

    def _build_llm_agent(self) -> LlmAgent:
        # LLM 的指令仍然保持不變
        personality_description = (
            f"You are an investment advisor with the following personality traits:\n"
            f"- Openness: {self.personality['openness']}\n"
            f"- Conscientiousness: {self.personality['conscientiousness']}\n"
            f"- Extraversion: {self.personality['extraversion']}\n"
            f"- Agreeableness: {self.personality['agreeableness']}\n"
            f"- Neuroticism: {self.personality['neuroticism']}\n\n"
            "Use ONLY the provided market data and technical indicators from the registered agents.\n"
            "Do not create or call any new agents.\n"
            "After analyzing, decide: BUY, SELL, or HOLD, and explain briefly why your personality leads you to this decision.\n"
        )

        instruction = personality_description + """
        Your task:
        1. Analyze the stock using only the given market data and technical indicators.
        2. Decide action: BUY, SELL, or HOLD.
        3. Calculate the number of shares to buy/sell based on available cash and current holdings.
        4. Always output **two parts**:
            - Part 1: strict JSON (for machine processing)
            - Part 2: a natural language explanation
        Do not output anything else.
        """

        return LlmAgent(
            model='gemini-2.5-flash',
            name='investment_highC_agent',
            description='Makes personality-driven investment decisions based on stock indicators and market sentiment.',
            instruction=instruction,
            tools=[self.make_investment_decision],
        )

    
    async def call_agent(self, agent_name: str, method_name: str, **kwargs: Any) -> Any:
        try:
            target_agent = self.agents.get(agent_name)
            if not target_agent:
                available_agents = list(self.agents.keys())
                raise ValueError(
                    f"Agent '{agent_name}' not found in registry. "
                    f"Available agents: {available_agents}"
                )

            method = getattr(target_agent, method_name, None)
            if not method:
                available_methods = [m for m in dir(target_agent) if not m.startswith('_')]
                raise AttributeError(
                    f"Method '{method_name}' not found on agent '{agent_name}'. "
                    f"Available methods: {available_methods}"
                )

            result = None
            if callable(method):
                if hasattr(method, '__await__'):
                    result = await method(**kwargs)
                else:
                    result = method(**kwargs)

            print(f"[DEBUG] {agent_name}.{method_name} returned: {result}")
            return result

        except (ValueError, AttributeError, TypeError) as e:
            logging.error(f"Failed to call agent '{agent_name}': {e}")
            print(f"[ERROR] Failed to call agent '{agent_name}': {e}")
            return {"error": str(e)}


    # 這是新的核心方法，它是一個高階的協調器
    async def make_investment_decision(self, stock_symbol: str) -> dict[str, Any]:
        """
        Coordinates with other agents to make a final investment decision.
        """
        # Step 1: 拿市場價格
        market_data_result = await self.call_agent("google_adk_market", "fetch_stock_data", symbol=stock_symbol)
        print("=== DEBUG: MarketDataAgent result ===")
        print(market_data_result)
        print("=====================================")

        # Step 2: 拿技術指標
        indicators_result = await self.call_agent("google_adk_stockIndicator", "analyze_indicators", symbol=stock_symbol)
        print("=== DEBUG: StockIndicatorAgent result ===")
        print(indicators_result)
        print("=========================================")

        # Step 3: 檢查是否有錯誤
        if 'error' in market_data_result or 'error' in indicators_result:
            return {"status": "error", "message": "Failed to get data from one of the agents."}

        # Step 4: 將代理人資料結構化成 JSON message，傳給 LLM
        llm_input_messages = [
            {
                "role": "system",
                "content": (
                    f"You are an investment advisor with the following personality traits:\n"
                    f"- Openness: {self.personality['openness']}\n"
                    f"- Conscientiousness: {self.personality['conscientiousness']}\n"
                    f"- Extraversion: {self.personality['extraversion']}\n"
                    f"- Agreeableness: {self.personality['agreeableness']}\n"
                    f"- Neuroticism: {self.personality['neuroticism']}\n\n"
                    "Use the provided market data and technical indicators from other agents to make a BUY, SELL, or HOLD decision. "
                    "Output strict JSON for machine processing and a natural language explanation."
                )
            },
            {
                "role": "user",
                "content": json.dumps({
                    "stock_symbol": stock_symbol,
                    "market_data": market_data_result,
                    "technical_indicators": indicators_result
                }, indent=2)
            }
        ]

        # Step 5: 呼叫 LLM，讓它使用代理人的資料
        decision_text = await self._runner.run(messages=llm_input_messages)
        print("=== DEBUG: LLM OUTPUT ===")
        print(decision_text)
        print("=====================================")

        # Step 6: 嘗試解析 JSON
        try:
            decision_json = json.loads(decision_text)
        except json.JSONDecodeError:
            decision_json = {
                "action": "HOLD",
                "reason": "Fallback default due to invalid JSON from LLM."
            }

        return {
            "status": "success",
            "decision": decision_json,
            "price": market_data_result.get("price"),
            "indicators": indicators_result.get("indicators"),
        }

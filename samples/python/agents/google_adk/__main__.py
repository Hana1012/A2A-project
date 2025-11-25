import logging
import os

import click

from agent import InvestmentAgent
from common.server import A2AServer
from common.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    MissingAPIKeyError,
)
from dotenv import load_dotenv
from task_manager import AgentTaskManager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', default='localhost')
@click.option('--port', default=10002)
def main(host, port):
    try:
        # Check for API key only if Vertex AI is not configured
        if not os.getenv('GOOGLE_GENAI_USE_VERTEXAI') == 'TRUE':
            if not os.getenv('GOOGLE_API_KEY'):
                raise MissingAPIKeyError(
                    'GOOGLE_API_KEY environment variable not set and GOOGLE_GENAI_USE_VERTEXAI is not TRUE.'
                )

        capabilities = AgentCapabilities(streaming=True)

        # Define agent skill 定義技能
        skill = AgentSkill(
            id='simulate_big5_daily_portfolio',
            name='High Neuroticism Daily Portfolio Simulation (高神經質每日投資組合模擬)',
            description=(
                'Simulates a daily multi-stock investment portfolio based on High Neuroticism Big Five personality traits. '
                'The agent allocates initial cash, decides which stocks to buy, sell, or hold each day, '
                'and adjusts the portfolio according to market data and technical indicators.\n'
                '模擬每日多股票投資組合，根據高開放性大五人格特質分配初始資金，'
                '每天根據市場數據和技術指標決定買入、賣出或持有股票，並調整投資組合。'
            ),
            tags=['investment', 'stock', 'big5', 'finance', 'daily', 'portfolio', 'simulate'],
            examples=[
                '初始資金 $100,000，模擬從 2024-01-01 到 2024-06-30 的每日投資決策。',
                '假設初始現金 $50,000，模擬 2024 年第一季的每日股票投資組合調整。',
                '使用 $200,000 初始資金，模擬 Tesla、Apple、Amazon 股票每日投資行為。',
                '模擬台積電（TSM）股票每日操作策略，期間為 2024 年第一季。'
            ]
        )

        # Create agent card 創建代理卡片
        agent_card = AgentCard(
            name='High Neuroticism Daily Investment Agent (高神經質每日投資代理)',
            description=(
                "This agent **simulates daily multi-stock investment portfolios** based on a user's High Openness (Big Five) personality traits.  \n"
                "It evaluates **market data**, **technical indicators** (e.g., MACD, RSI, volatility),  \n"
                "and **decides daily** which stocks to buy, sell, or hold.  \n"
                "Initial cash is provided, and the portfolio is updated day by day without additional funding.  \n\n"
                "**中文說明**  \n"
                "此代理會根據用戶高開放性人格特質模擬每日多股票投資組合，結合市場數據和技術指標（如 MACD、RSI、波動率），  \n"
                "每天決定買入、賣出或持有股票，初始資金由用戶提供，後續投資組合每日更新，不額外增加資金。"
            ),
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=InvestmentAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=InvestmentAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # 啟動 A2A server
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=InvestmentAgent(personality={"openness": "high"})),
            host=host,
            port=port,
        )
        server.start()

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main()

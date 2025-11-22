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
            id='simulate_big5_investor',
            name='High Openness (Big Five) Investor Simulation (高開放性（大五人格）投資者模擬)',
            description=(
                'Simulates the investment behavior of a stock investor based on High Openness Big Five personality traits, '
                'and decides whether to buy, sell, or hold a stock using technical indicators and market trends. \n'
                '模擬具有高開放性大五人格特質的股票投資者，結合技術指標和市場趨勢，判斷買入、賣出或持有股票。'
            ),
            tags=['investment', 'stock', 'big5', 'finance', 'decision-making', 'simulate'],
            examples=[
                '目前有美金$100,000，請分析 Google 的股票，並模擬從 2024 年 1 月 1 日到 2024 年 6 月 30 日的投資。',
                '模擬投資 Tesla 股票，時間區間為 2023 年全年，請給出買進、持有或賣出的決策過程。',
                '請分析 Apple 股票在 2022 年到 2023 年的走勢，並模擬投資策略。',
                '假設投資資金為 $50,000，模擬 2024 年第一季到第二季投資 Microsoft 股票的決策。',
                '若持有 Meta 股票，請模擬從 2023 年 7 月到 2023 年 12 月的投資決策與可能的報酬。',
                '請模擬投資 Nvidia 股票，在 2024 年全年的操作策略與風險控制。',
                '分析 Amazon 股票於 2023 年的市場表現，並模擬投資策略。',
                '請針對台積電（TSM）股票模擬 2024 年第一季的投資決策。'
            ]
        )


        # Create agent card 創建代理卡片
        agent_card = AgentCard(
            name='High OpennessInvestment Decision Agent (高開放性人格投資決策代理)',
            description=(
                "This agent **simulates stock investment decisions** based on a user's High openness (Big Five) personality traits.  \n"
                "It evaluates **stock trends**, **technical indicators** (e.g., MACD, RSI, volatility),  \n"
                "and **makes concrete decisions**: buy, sell, or hold, reflecting the investor's risk tolerance and personality.  \n\n"
                "**中文說明**  \n"
                "此代理會根據用戶的高開放性人格模擬股票投資決策，結合股價趨勢、技術指標（如 MACD、RSI、波動率）  \n"
                "給出具體操作建議：買入、賣出或持有，反映該投資者的風險偏好與性格特質。"
            ),
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=InvestmentAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=InvestmentAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=InvestmentAgent()),
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

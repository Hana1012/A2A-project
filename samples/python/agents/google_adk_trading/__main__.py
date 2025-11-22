import logging
import os

import click
from dotenv import load_dotenv

from agent import MarketAgent  # 改成 MarketAgent
from common.server import A2AServer
from common.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    MissingAPIKeyError,
)
from task_manager import AgentTaskManager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', default='localhost')
@click.option('--port', default=10012)  # 改一個新的 port
def main(host, port):
    try:
        # 如果未使用 Vertex AI，則需檢查 API 金鑰
        if not os.getenv('GOOGLE_GENAI_USE_VERTEXAI') == 'TRUE':
            if not os.getenv('GOOGLE_API_KEY'):
                raise MissingAPIKeyError(
                    'GOOGLE_API_KEY environment variable not set and GOOGLE_GENAI_USE_VERTEXAI is not TRUE.'
                )

        capabilities = AgentCapabilities(streaming=True)
        skill = AgentSkill(
            id='simulate_market_trading',
            name='Simulate Market Trading',
            description=(
                "Simulate a stock exchange order book. \n"
                "Accepts buy/sell orders with symbol, price, and quantity. \n"
                "Matches orders automatically if prices overlap (buy >= sell). \n"
                "Provides execution reports and current order book status."
            ),
            tags=[
                'market',
                'finance',
                'trading',
                'order-book',
                'simulation',
                'buy',
                'sell',
                'matching-engine',
            ],
            examples=[
                'Place a buy order for AAPL at 180.5 with quantity 10',
                'Sell 5 shares of TSLA at 250.0',
                'Show me the current order book for MSFT',
            ],
        )

        agent_card = AgentCard(
            name='Market Agent',
            description=(
                "An agent that simulates a stock exchange market.  \n"
                "It maintains an order book, accepts buy/sell orders, \n"
                "matches them automatically, and reports trades and current market depth."
            ),
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=MarketAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=MarketAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=MarketAgent()),
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

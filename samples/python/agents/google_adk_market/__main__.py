import logging
import os

import click
from dotenv import load_dotenv

from agent import MarketDataAgent 
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
            id='provide_stock_data',
            name='Provide Stock Data',
            description=(
                "Fetch historical or real-time stock prices. \n"
                "Accepts a stock symbol and a date range. \n"
                "Returns a list of (date, open, close) prices."
            ),
            tags=[
                'market',
                'finance',
                'data',
            ],
            examples=[
                'Get AAPL stock data from 2024-01-01 to 2024-01-10',
                'Show me TSLA prices for the last 7 days',
                'Fetch MSFT stock open and close between 2024-05-01 and 2024-05-15',
            ],
        )

        agent_card = AgentCard(
            name='Market Data Agent',
            description=(
                "An agent that provides stock market data.  \n"
                "It fetches historical or real-time prices given a stock symbol and date range, \n"
                "returning open/close price information for analysis."
            ),
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=MarketDataAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=MarketDataAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )


        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=MarketDataAgent()),
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

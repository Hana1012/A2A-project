import logging
import os

import click
from dotenv import load_dotenv

from agent import StockTimeSeriesAgent  
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
@click.option('--port', default=10010)  
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
            id='analyze_stock_trend',
            name='Analyze Stock Trend',
            description='Analyze recent stock price trends using time series data.\n Provides insights such as trend direction, volatility, and average daily change.',
            tags=['stock', 'finance', 'time series', 'trend'],
            examples=[
                '請分析 AAPL 最近的股價趨勢',
                'What is the volatility of TSLA stock over the past 10 days?',
            ],
        )
        agent_card = AgentCard(
            name='Stock Time Series Agent',
            description='Agent that analyzes stock prices using time series data and provides technical insights.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=StockTimeSeriesAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=StockTimeSeriesAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=StockTimeSeriesAgent()),
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

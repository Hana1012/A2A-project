import logging
import os

import click
from dotenv import load_dotenv

from agent import StockIndicatorAgent  
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
@click.option('--port', default=10011)  
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
            id='analyze_stock_indicators',
            name='Analyze Stock Indicators & Market Sentiment',
            description=(
                "Analyze stock prices using historical time series data. \n"
                "Calculates key investment indicators such as Simple Moving Average (SMA), \n"
                "Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), \n"
                "Bollinger Bands, GARCH(1,1) volatility forecast, and price volatility. \n"
                "Integrates market news sentiment to provide actionable insights into trend, momentum, risk, \n"
                "and potential support/resistance levels."
            ),
            tags=[
                'stock',
                'finance',
                'technical-analysis',
                'trend',
                'SMA',
                'RSI',
                'MACD',
                'Bollinger Bands',
                'volatility',
                'GARCH',
                'news',
                'sentiment',
                'investment'
            ],
            examples=[
                '請幫我分析 AAPL 在 2024-08-01 到 2024-08-31 的技術指標、GARCH 波動率與市場情緒',
                'Show me RSI, MACD, GARCH volatility, and news sentiment for TSLA stock in the past month.',
                'What are the Bollinger Bands, GARCH forecast, and news sentiment for MSFT this quarter?',
            ],
        )

        agent_card = AgentCard(
            name='Stock Indicator & Market Sentiment Agent',
            description=(
                "An agent that analyzes stock prices using historical time series data, \n"
                "calculates key investment indicators (SMA, RSI, MACD, Bollinger Bands, GARCH volatility, and price volatility), \n"
                "and integrates market news sentiment. \n"
                "Provides actionable insights for short-term and long-term trading decisions, \n"
                "including trend, momentum, risk, and potential support/resistance levels."
            ),
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=StockIndicatorAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=StockIndicatorAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )


        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=StockIndicatorAgent()),
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

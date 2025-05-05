# === main.py ===

import click, logging

from agent import UtilityAgent
from task_manager import AgentTaskManager
from common.server.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=8002)


def main(host, port):
    
    try:
         
        capabilities = AgentCapabilities(streaming=True)

        skills = [
            AgentSkill(
                id="utility_agent",
                name="Utility Agent",
                description="Answers weather and time-related questions using external tools.",
                tags=["weather", "time"],
                examples=[
                    "What time is it in Brazil?", "What is the weather like now?"
                    ]
                )
            ]

        agent_card = AgentCard(
            name="Utility Agent",
            description="Answers weather and time-related queries using tool results.",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=UtilityAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=UtilityAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=skills,
        )
        print(AgentTaskManager())
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=UtilityAgent()),
            host=host,
            port=port
        )

        logger.info(f"Starting Utilities MCP agent on {host}:{port}")
        server.start()
        logger.info(f"✅ Uvicorn should now be running at http://{host}:{port}")

    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        exit(1)


if __name__ == "__main__":
        main()

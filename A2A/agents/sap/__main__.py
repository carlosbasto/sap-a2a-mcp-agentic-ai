import click
import logging

from agent import SAPAgent
from task_manager import AgentTaskManager
from common.server.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=8001)

def main(host, port):
    """Starts the SAP Retriever Agent via A2A server."""

    try:
        capabilities = AgentCapabilities(streaming=True)

        skills = [
            AgentSkill(
                id="sap_agent",
                name="SAP Agent",
                description="Answers SAP-related questions using a retriever tool.",
                tags=["sap", "retriever", "knowledge"],
                examples=[
                    "What is SAP Datasphere?",
                    "Explain the role of SAP BTP.",
                    "How does SAP S/4HANA differ from ECC?"
                ]
            )
        ]

        agent_card = AgentCard(
            name="SAP Agent",
            description="Intelligent agent for answering SAP-related questions using knowledge retrieval.",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=SAPAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=SAPAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=skills,
        )

        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=SAPAgent()),
            host=host,
            port=port
        )

        logger.info(f"Starting SAP Retriever Agent on {host}:{port}")
        server.start()
        logger.info(f"Uvicorn should now be running at http://{host}:{port}")

    except MissingAPIKeyError as e:
        logger.error(f"API Key missing: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Startup error: {e}")
        exit(1)


if __name__ == "__main__":
    main()

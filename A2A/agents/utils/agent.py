import json
import asyncio
from typing import AsyncIterable, Dict, Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage, AssistantMessage
from gen_ai_hub.orchestration.models.template import Template
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.service import OrchestrationService
from gen_ai_hub.orchestration.models.response_format import ResponseFormatJsonSchema


class UtilityAgent:
    """Agent that uses tool-calling reasoning via SAP GenAI Hub and MCP session."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        pass

    def _build_dynamic_schema(self) -> dict:
        """Define expected JSON schema for tool-calling reasoning."""
        return {
            "title": "ToolCalls",
            "type": "object",
            "properties": {
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "decision": {"type": "string"},
                            "reason": {"type": "string"},
                            "function": {"type": "string"},
                            "parameters": {"type": "object"},
                        },
                        "required": ["decision", "reason", "function", "parameters"],
                    }
                }
            },
            "required": ["tool_calls"]
        }

    async def _generate_instruction(self) -> str:
        """Generate dynamic system prompt based on registered tools."""
        description = json.dumps(await self.list_tools(), indent=2)
        return f"""
        You are an intelligent AI assistant capable of deciding whether to invoke tools based on the user's request.

        Available tools:
        {description}

        Instructions:
        - For each relevant tool, return a JSON entry with the function name and parameters.
        - If no tool is relevant, return an entry with decision = "no_tool".

        Return ONLY valid JSON like:
        {{
          "tool_calls": [
            {{
              "decision": "tool",
              "reason": "The user asked for weather.",
              "function": "get_weather",
              "parameters": {{
                "latitude": 48.8566,
                "longitude": 2.3522
              }}
            }},
            {{
              "decision": "tool",
              "reason": "The user asked for time.",
              "function": "get_time_now",
              "parameters": {{}}
            }}
          ]
        }}
        """

    async def list_tools(self) -> dict:
        """Fetch list of tools from the MCP session."""
        tools_result = await self.session.list_tools()
        return {tool.name: {"description": tool.description} for tool in tools_result.tools}

    async def _execute_tool(self, decision: dict) -> str:
        """Execute a single tool decision."""
        try:
            result = await self.session.call_tool(decision["function"], arguments=decision.get("parameters", {}))
            return result.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"

    async def _finalize_response(self, user_query: str, tool_results: list, messages: list) -> str:
        """Generate final LLM answer after tools were executed."""
        messages.append(SystemMessage(
            """
            You now have access to the results provided by the tools. When the results are clear and complete, use only that information to answer the user's 
            question in a natural, helpful, and concise manner. However, if any result appears vague, incomplete, or states uncertainty (e.g., 'I don't know'), 
            rely on your own knowledge to deliver an accurate and informative response.
            Always avoid requesting information already provided. Focus on clarity, relevance, and user value.
            """
        ))

        messages.append(UserMessage(f"User question: {user_query}"))

        summary = "\n".join([f"- Tool `{name}` returned: {json.dumps(result)}" for name, result in tool_results])
        messages.append(UserMessage(f"Tool Results:\n{summary}"))

        template = Template(messages=messages, response_format="text")
        config = OrchestrationConfig(template=template, llm=self.llm)
        response = OrchestrationService(config=config).run()
        return response.module_results.llm.choices[0].message.content

    async def run(self, user_query: str) -> str:
        """Main method to orchestrate reasoning and tool invocation."""
        system_message = SystemMessage(await self._generate_instruction())
        prompt = UserMessage(user_query)
        messages = [system_message, prompt]

        template = Template(
            messages=messages,
            response_format=ResponseFormatJsonSchema(
                name="ToolCall",
                description="Tool execution format",
                schema=self._build_dynamic_schema()
            )
        )

        config = OrchestrationConfig(template=template, llm=self.llm)
        response = OrchestrationService(config=config).run()

        decisions_json = json.loads(response.module_results.llm.choices[0].message.content)
        tool_results = []

        for decision in decisions_json.get("tool_calls", []):
            if decision.get("decision") == "tool":
                tool_response = await self._execute_tool(decision)
                tool_results.append((decision["function"], tool_response))
                messages.append(AssistantMessage(json.dumps(decision)))
            else:
                messages.append(AssistantMessage(json.dumps(decision)))

        return await self._finalize_response(user_query, tool_results, messages)

    async def stream(self, query: str, sessionId: str) -> AsyncIterable[Dict[str, Any]]:
        """
        Entry point for streaming response to A2A task manager.
        Initializes MCP session and streams a final task result.
        """
        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "Processing your request..."
        }

        try:
            async with sse_client("https://mcp-server-sap.c-49e33bd.stage.kyma.ondemand.com/sse") as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    self.llm = LLM(name="gpt-4o", version="latest", parameters={"max_tokens": 2000, "temperature": 0.2})
                    response_text = await self.run(query)

            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": response_text
            }

        except Exception as e:
            yield {
                "is_task_complete": True,
                "require_user_input": True,
                "content": f"An error occurred: {str(e)}"
            }

    def invoke(self, query: str, sessionId: str) -> Dict[str, Any]:
        """Not implemented — this agent supports streaming only."""
        raise NotImplementedError("Use the streaming interface (tasks/sendSubscribe) instead.")

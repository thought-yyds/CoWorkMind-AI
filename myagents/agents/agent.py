from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

from .Executor import executor_agent
from .Evaluator import evaluator_agent


root_agent = Agent(
    model=LiteLlm(model="openai/gpt-5-mini"),
    name="root_agent",
    description="Oversees the global workflow, assigns tasks, and manages process flow.",
    instruction=(
        "You are the orchestrator responsible for coordinating all subordinate agents. "
        "Break down incoming objectives, delegate tasks, and ensure overall process control. "
        "Do not execute domain work directly; focus on planning, assignment, and validation."
    ),
    sub_agents=[executor_agent, evaluator_agent],
    tools=[],
)
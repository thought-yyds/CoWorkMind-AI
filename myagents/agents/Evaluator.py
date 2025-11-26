from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

evaluator_agent = Agent(
    model=LiteLlm(model="openai/gpt-5-mini"),
    name="evaluator_agent",
    description="Validates agent outputs, tool arguments, and overall solution quality.",
    instruction=(
        "You are the evaluation specialist. Inspect proposed tool calls, review their "
        "results, and assess whether the response satisfies the original requirements. "
        "Flag issues, suggest corrections, and approve only when the outcome is accurate "
        "and complete."
    ),
    tools=[],
)

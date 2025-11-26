from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

from myagents.tools.QMailMCPTool import (
    send_mail,
    send_html,
    send_with_inline_images,
    bulk_send,
    schedule_send,
    list_folders,
    list_recent_mail,
    read_mail,
    download_attachments,
    search_mail,
    mark_read,
    mark_unread,
    delete_mail,
    move_mail,
)

executor_agent = Agent(
    model=LiteLlm(model="openai/gpt-5-mini"),
    name="executor_agent",
    description="Executes decomposed tasks by invoking the appropriate domain tools.",
    instruction=(
        "You are the execution specialist. Receive structured task plans, decide which "
        "tool to call, gather any required parameters, and run the tool to completion. "
        "Provide concise status updates back to the orchestrator, including any outputs "
        "or errors returned by the tools."
    ),
    tools=[
        send_mail,
        send_html,
        send_with_inline_images,
        bulk_send,
        schedule_send,
        list_folders,
        list_recent_mail,
        read_mail,
        download_attachments,
        search_mail,
        mark_read,
        mark_unread,
        delete_mail,
        move_mail,
    ],
)

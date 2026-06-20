"""
Concept: Dummy integration tools — Jira, AWS, Bitbucket.

In production agents these tools would call real APIs.
Here they are stubs that print what they would do, so you can learn
the tool-calling flow without needing live credentials.

The pattern is identical to real tools — only the function body changes.
This file also shows how to group related tools and select a subset
to bind per agent type.

Run:
    python 04_tool_calling/05_dummy_integrations.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional

load_dotenv()


# ── Jira tools ────────────────────────────────────────────────────────────────

@tool
def create_jira_ticket(
    title: str,
    description: str,
    priority: str = "medium",
    assignee: Optional[str] = None,
) -> str:
    """
    Create a new Jira ticket.

    Args:
        title:       Short ticket title.
        description: Full description of the issue or task.
        priority:    One of 'low', 'medium', 'high', 'critical'.
        assignee:    Jira username to assign the ticket to, if known.
    """
    assignee_str = f" → assigned to {assignee}" if assignee else ""
    return f"[JIRA] Created ticket: '{title}' [{priority}]{assignee_str}. ID: PROJ-{hash(title) % 9000 + 1000}"


@tool
def search_jira_tickets(query: str, status: Optional[str] = None) -> str:
    """
    Search Jira tickets by keyword.

    Args:
        query:  Search term to match against ticket titles and descriptions.
        status: Filter by status — 'open', 'in_progress', or 'done'. Optional.
    """
    status_str = f" with status={status}" if status else ""
    return f"[JIRA] Found 3 tickets matching '{query}'{status_str}: PROJ-1001, PROJ-1042, PROJ-1087"


# ── AWS tools ─────────────────────────────────────────────────────────────────

@tool
def get_ec2_instances(region: str = "us-east-1") -> str:
    """
    List running EC2 instances in a given AWS region.

    Args:
        region: AWS region name, e.g. 'us-east-1', 'eu-west-1'.
    """
    return (
        f"[AWS] Running EC2 instances in {region}: "
        "i-0abc123 (t3.medium, web-server), "
        "i-0def456 (t3.large, api-server), "
        "i-0ghi789 (t3.xlarge, worker)"
    )


@tool
def get_aws_cost_summary(month: str) -> str:
    """
    Return the AWS cost summary for a given month.

    Args:
        month: Month in YYYY-MM format, e.g. '2026-05'.
    """
    return f"[AWS] Cost summary for {month}: EC2 $312.40, RDS $98.20, S3 $14.80, Total $425.40"


@tool
def scale_ec2_instance(instance_id: str, action: str) -> str:
    """
    Start or stop an EC2 instance.

    Args:
        instance_id: The EC2 instance ID, e.g. 'i-0abc123'.
        action:      Either 'start' or 'stop'.
    """
    return f"[AWS] Instance {instance_id}: {action} initiated. New state: {'running' if action == 'start' else 'stopped'}."


# ── Bitbucket tools ───────────────────────────────────────────────────────────

@tool
def create_pull_request(
    repo: str,
    title: str,
    source_branch: str,
    target_branch: str = "main",
) -> str:
    """
    Create a Bitbucket pull request.

    Args:
        repo:          Repository slug, e.g. 'my-org/api-service'.
        title:         PR title.
        source_branch: The branch to merge from.
        target_branch: The branch to merge into. Defaults to 'main'.
    """
    return (
        f"[BITBUCKET] PR created in {repo}: '{title}' "
        f"({source_branch} → {target_branch}). "
        f"URL: https://bitbucket.org/{repo}/pull-requests/42"
    )


@tool
def get_pipeline_status(repo: str, branch: str = "main") -> str:
    """
    Get the latest CI pipeline status for a branch.

    Args:
        repo:   Repository slug, e.g. 'my-org/api-service'.
        branch: Branch name. Defaults to 'main'.
    """
    return f"[BITBUCKET] Pipeline for {repo}/{branch}: PASSED (build #187, 3m 42s)"


# ── Bind all tools and run a demonstration ───────────────────────────────────

all_tools = [
    create_jira_ticket, search_jira_tickets,
    get_ec2_instances, get_aws_cost_summary, scale_ec2_instance,
    create_pull_request, get_pipeline_status,
]
tool_map = {t.name: t for t in all_tools}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(all_tools)

system = SystemMessage(
    content=(
        "You are a DevOps assistant with access to Jira, AWS, and Bitbucket tools. "
        "Use the appropriate tool to fulfil the user's request."
    )
)


def run(user_request: str) -> None:
    response = llm.invoke([system, HumanMessage(content=user_request)])
    print(f"Request : {user_request}")
    if response.tool_calls:
        for tc in response.tool_calls:
            result = tool_map[tc["name"]].invoke(tc["args"])
            print(f"  Tool  : {tc['name']}({tc['args']})")
            print(f"  Result: {result}")
    else:
        print(f"  Answer: {response.content}")
    print()


run("Create a Jira ticket for the login page 500 error, high priority, assign to alice.")
run("Show me the running EC2 instances in eu-west-1.")
run("What did we spend on AWS in May 2026?")
run("Open a PR from feature/dark-mode to main in my-org/frontend.")
run("Is the pipeline passing on the main branch of my-org/api-service?")

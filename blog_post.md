# Mitigating LLM Hallucinations with a Self-Correcting Multi-Agent System on SAP AI Core

## Introduction

Large Language Models (LLMs) have demonstrated incredible capabilities in understanding and generating human-like text, powering a new generation of AI-driven applications. However, one of the most significant challenges that persists is the phenomenon of "hallucination," where models generate information that is factually incorrect, nonsensical, or untethered from the input context. While LLMs are becoming more powerful, ensuring the reliability and accuracy of their outputs remains a critical hurdle for enterprise adoption.

To address this challenge, we can move beyond single-model architectures and embrace more sophisticated, multi-agent systems. This blog post introduces a powerful approach: a self-correcting, two-agent system built on **SAP AI Core**. This system is designed to significantly reduce hallucinations by introducing a layer of automated verification and refinement.

The core idea is simple yet effective:
1.  A **Worker Agent** is responsible for executing the primary task, such as answering a complex question or performing research.
2.  An **Evaluator Agent** scrutinizes the Worker's response against a set of success criteria. It provides feedback, requests clarification, or flags the response for user review if it's not up to par.

This continuous feedback loop, where one agent's output is another agent's input, creates a "self-correcting" mechanism. By having one agent monitor another's response and ask specific questions, we can catch inaccuracies and refine answers before they reach the end-user. This not only improves the quality and reliability of the final output but also builds greater trust in AI-powered solutions.

In this technical guide, we will walk you through the step-by-step process of building such a system using the code from this repository. We'll leverage the power of SAP AI Core for robust model deployment, LangGraph for creating stateful agent workflows, and powerful LLMs to bring our agents to life. Let's dive in!

## The Two-Agent Architecture

Our system is built around a two-agent collaboration, where each agent has a distinct and specialized role. This separation of concerns is the key to improving the quality and reliability of the final output.

![Graph Visualization](https://mermaid.ink/img/pako:eNplkMFqwzAMhl_F6CmS-wAdOuyyMWAPpT1kpa21h2VJV8lK0H_faSoHnYAwSUL-_byXyCGoUeF6H2o0H4Wz0Gk6G9jQ53K6qW9z1F5sU4f2pT4x_kY-X6wT62v5e5aW61P-9n57KavXqLd8gH_Q6P3kQ-614w6k9JpWz0xYp1hQyB2mN28yK7hQxJmB0R78hM2Q3fD04Y7i9Xb4y7G10y64-7O-E56524s7eM7y3g5L4K9-Q5n6G3z)

*A visualization of the agent graph, showing the flow between the Worker, Tools, and Evaluator nodes.*

### 1. The Worker Agent: The Task Handler

The **Worker Agent** is the primary actor responsible for handling the user's request. In our implementation, this agent is powered by a capable model like **GPT-4.1**, which is well-suited for complex reasoning and tool usage.

The Worker's responsibilities include:
-   **Understanding the Task**: It receives the user's prompt and a set of "success criteria" that define what a successful outcome looks like.
-   **Using Tools**: The agent is equipped with tools, such as web browsing capabilities (via Playwright), to gather external information. This allows it to answer questions about recent events or access information beyond its training data.
-   **Maintaining Context**: It keeps track of the conversation history to provide coherent, multi-turn responses.
-   **Formulating a Response**: Based on the prompt, its internal knowledge, and any information gathered from tools, it generates a response.

Crucially, the Worker operates in a loop. If its initial response is deemed insufficient by the Evaluator, it receives feedback and retries the task, incorporating the suggestions to improve its next attempt.

### 2. The Evaluator Agent: The Quality Gatekeeper

The **Evaluator Agent** acts as a critical quality gate. Its sole purpose is to review the Worker Agent's output and determine if it meets the predefined success criteria. We also use a powerful model like **GPT-4.1** for this agent, as it needs to perform nuanced assessment.

The Evaluator's workflow is as follows:
-   **Receives the Response**: It takes the Worker's final response and the full conversation history as input.
-   **Assesses Against Criteria**: It compares the response against the user-defined success criteria.
-   **Generates Structured Feedback**: The Evaluator produces a JSON object containing three key fields:
    -   `feedback`: A human-readable explanation of what the Worker did well or where it fell short.
    -   `success_criteria_met`: A boolean flag indicating if the task is complete.
    -   `user_input_needed`: A boolean flag to determine if the Worker is stuck and requires clarification from the user.

### How This Architecture Reduces Hallucination

The magic of this system lies in the interaction between the two agents. Hallucinations often occur when a single LLM generates a plausible-sounding but incorrect answer without any checks and balances. Our two-agent system introduces several layers of defense:

-   **Explicit Success Criteria**: By forcing the user to define what success looks like, we provide the Evaluator with a clear benchmark.
-   **Automated Verification**: The Evaluator acts as an automated "second opinion." It is prompted to be critical and to verify the Worker's claims against the success criteria.
-   **Iterative Refinement**: If the Worker's response is inaccurate, the Evaluator sends it back with specific feedback. This forces the Worker to reconsider its approach, use its tools again, or try a different line of reasoning.
-   **Human in the Loop**: If the system gets stuck in a loop or the Worker is unable to satisfy the criteria, the Evaluator can flag that user input is needed. This prevents the agent from going down a rabbit hole of incorrect responses and allows the user to provide clarification.

This self-correcting loop, managed by a stateful graph, ensures that responses are more accurate, well-researched, and aligned with the user's intent, thereby building a more reliable and trustworthy AI assistant.

## Step-by-Step Implementation Guide

Now, let's get our hands dirty and build this system. The implementation is done within a Jupyter Notebook (`main.ipynb`) for easy, interactive development.

### Step 1: Prerequisites and Setup

First, we need to install the necessary Python packages and set up our environment.

**1.1. Install Dependencies**

The core components we'll be using are:
-   `langgraph`: To build our stateful, multi-agent graph.
-   `langchain-community`, `langchain-openai`: For agentic components and toolkits.
-   `langsmith`: For tracing and monitoring our agent's behavior.
-   `gradio`: To create a simple web interface for interaction.
-   `playwright`: For web browsing tools.

You can install them all with the following command:

```bash
pip install "langgraph[all]" langchain-community langchain-openai langsmith gradio playwright
python -m playwright install
```

**1.2. Environment Variables**

Create a `.env` file in your project root to store your SAP AI Core credentials securely. This is crucial for connecting to the service.

```env
AIC_CLIENT_ID=your_client_id
AIC_CLIENT_SECRET=your_client_secret
AIC_RESOURCE_GROUP=your_resource_group
AIC_BASE_URL=your_base_url
AIC_AUTH_URL=your_auth_url
AIC_FOUNDATIN_MODEL=gpt-4
AIC_ORCH_URL=your_orchestration_url
AIC_ORCH_MODEL=your_orchestration_model
LANGCHAIN_TRACING_V2=true
LANGSMITH_PROJECT=your_project_name
```

**1.3. Initial Imports and Setup**

In the notebook, we import all the necessary modules and load the environment variables. We also initialize the LangSmith client for tracing.

```python
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Configure LangSmith
# ... LangSmith client initialization ...
```

### Step 2: Schema and Tool Configuration

We define the data structures for our agent's state and configure the tools it will use.

**2.1. State Schema**

We use `TypedDict` to define the state of our graph. This state is passed between nodes and updated at each step. It includes the conversation messages, success criteria, and feedback.

```python
class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool
```

**2.2. Configure Web Browsing Tools**

We use `Playwright` to give our Worker Agent the ability to browse the web.

```python
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser

async_browser = create_async_playwright_browser(headless=False)
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()
```

### Step 3: SAP AI Core Integration

This is where we connect to SAP AI Core to use its powerful LLMs. The `llm_client.py` file contains two classes, `CL_Foundation_Service` and `CL_Orchestration_Service`, to simplify interactions.

**3.1. Initialize LLM Clients**

We create instances of our service classes and use them to get configured LLM clients. We use `gpt-4.1` for both the Worker and the Evaluator agent.

```python
from llm_client import CL_Foundation_Service

# Configure SAP AI Core
aic_config = {
    "aic_client_id": os.getenv('AIC_CLIENT_ID'),
    # ... other credentials
}

foundation_service = CL_Foundation_Service(aic_config)

# Worker agent uses gpt-4.1
worker_llm = foundation_service.get_llm_client(model_name='gpt-4.1')
llm_with_tools = worker_llm.bind_tools(tools)

# Evaluator agent also uses gpt-4.1
evaluator_llm = foundation_service.get_llm_client(model_name='gpt-4.1')
```

### Step 4: Agent and Graph Implementation

Now we define the logic for our agents and wire them together in a graph.

**4.1. Define Agent Nodes**

We create Python functions that will serve as the nodes in our graph. Each function takes the current `State` as input and returns an updated state.

-   **`worker(state: State)`**: This function contains the logic for the Worker Agent. It constructs a system prompt, includes any feedback from the Evaluator, and invokes the LLM with the available tools.
-   **`evaluator(state: State)`**: This function holds the Evaluator Agent's logic. It assesses the Worker's response against the success criteria and returns a structured JSON object with its findings.

**4.2. Construct the Graph**

We use `StateGraph` from LangGraph to build the workflow.

```python
from langgraph.graph import StateGraph, END

graph_builder = StateGraph(State)

# Add nodes for the worker, evaluator, and tools
graph_builder.add_node("worker", worker)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_node("evaluator", evaluator)

# The graph starts with the worker
graph_builder.add_edge(START, "worker")

# Add conditional edges for routing logic
graph_builder.add_conditional_edges(
    "worker",
    worker_router, # A function that checks if the last message has tool calls
    {"tools": "tools", "evaluator": "evaluator"}
)
graph_builder.add_edge("tools", "worker") # After using tools, go back to the worker

graph_builder.add_conditional_edges(
    "evaluator",
    route_based_on_evaluation, # A function that checks if the task is done
    {"worker": "worker", "END": END}
)

# Compile the graph
graph = graph_builder.compile()
```
The `worker_router` decides whether to call the tools or pass the response to the evaluator. The `route_based_on_evaluation` function determines whether the task is complete or if the worker needs to try again.

### Step 5: Running the System and Testing

The repository includes a Gradio-based web interface and a suite of test cases to validate the system's functionality.

**5.1. Gradio Web Interface**

The `main.ipynb` notebook includes code to launch an interactive Gradio chat interface. You can input your request, define the success criteria, and see the multi-agent system in action.

**5.2. Test Cases**

The notebook also contains four test cases that demonstrate the system's capabilities:
1.  **Basic Human Interaction**: Tests self-awareness and basic response generation.
2.  **Web Research**: Validates the agent's ability to use tools to find recent information.
3.  **Feedback Loop**: Shows how the agent iteratively improves its response based on feedback.
4.  **Complex Multi-step Tasks**: A comprehensive test of the agent's ability to handle complex requests with multiple requirements.

Running these tests provides a clear picture of how the self-correcting mechanism works and how it can be monitored using LangSmith.

## Conclusion

In the rapidly evolving landscape of generative AI, tackling the challenge of LLM hallucinations is paramount for building enterprise-ready applications. The self-correcting, multi-agent architecture presented here offers a robust and practical solution. By creating a system of checks and balances with a Worker and an Evaluator agent, we can significantly improve the factual accuracy and overall quality of AI-generated responses.

The key advantages of this approach are:
-   **Enhanced Reliability**: The iterative feedback loop catches and corrects errors before they reach the user.
-   **Greater Accuracy**: The use of external tools for research, combined with critical evaluation, ensures that responses are grounded in verifiable information.
-   **Increased Trust**: By delivering more reliable outputs and providing transparency into the agent's reasoning process (via LangSmith), we can build greater user trust.

Powered by **SAP AI Core**, this architecture provides the scalability, security, and performance needed for enterprise use cases. The integration with open-source tools like LangGraph and LangChain makes it a flexible and powerful pattern for developers in the SAP ecosystem to adopt.

We encourage the SAP AI Core community to explore this self-correcting agent model. The code in this repository serves as a starting point for building your own advanced, reliable, and trustworthy AI assistants. As we continue to push the boundaries of what's possible with AI, these kinds of sophisticated agentic workflows will be essential in harnessing the full potential of Large Language Models responsibly.
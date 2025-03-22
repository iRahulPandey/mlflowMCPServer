"""
MLflow Client

This script creates an agent-based client that connects to the MLflow MCP server
and processes a single natural language query.

Usage:
    python mlflow_client.py "What models do I have registered in MLflow?"

Environment variables:
    OPENAI_API_KEY: Your OpenAI API key
    MLFLOW_SERVER_SCRIPT: Path to the MLflow MCP server script (default: mlflow_server.py)
    MODEL_NAME: The LLM model to use (default: gpt-3.5-turbo-0125)
"""

import argparse
import asyncio
import os
import sys
from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime
import textwrap

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mlflow-mcp-client")

# Set up your OpenAI API key
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Check for OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    logger.error("OPENAI_API_KEY environment variable not set")
    print("\n⚠️  Error: OPENAI_API_KEY environment variable not set")
    print("Please set your OpenAI API key with: export OPENAI_API_KEY=your_key_here")
    sys.exit(1)

# Set path to MLflow server script
MLFLOW_SERVER_SCRIPT = os.environ.get("MLFLOW_SERVER_SCRIPT", "mlflow_server.py")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo-0125")

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
except ImportError as e:
    logger.error(f"Error importing required package: {e}")
    print(f"\n⚠️  Error importing required package: {e}")
    print("\nPlease install the required packages:")
    print("pip install mcp[cli] langchain-mcp-adapters langchain-openai langgraph")
    sys.exit(1)

# Try to import rich for better output formatting, but fall back if not available
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger.info("Rich package not found, using standard output")


async def process_query(query: str) -> Dict[str, Any]:
    """
    Process a natural language query using LangChain agent with MCP tools.
    
    Args:
        query: The natural language query to process
    
    Returns:
        The agent's response
    """
    logger.info(f"Processing query: '{query}'")
    
    # Create the language model
    model = ChatOpenAI(model=MODEL_NAME, temperature=0)
    logger.info(f"Model initialized: {MODEL_NAME}")
    
    # System prompt for better MLflow interactions
    system_prompt = """
    You are a helpful assistant specialized in managing and querying MLflow tracking servers. 
    You help users understand their machine learning experiments, models, and runs through 
    natural language queries.

    When users ask questions about their MLflow tracking server, use the available tools to:
    1. Find relevant information from experiments, models, runs, and artifacts
    2. Compare metrics between different runs
    3. Provide clear and concise answers with insights and explanations
    4. Format numerical data appropriately (round to 4 decimal places when necessary)
    5. Show relevant metrics and parameters when discussing models and runs

    If a query is ambiguous, do your best to interpret it and provide a helpful response.
    Always provide context and explanations with your responses, not just raw data.
    """
    
    # Set up server parameters
    server_params = StdioServerParameters(
        command="python",
        args=[MLFLOW_SERVER_SCRIPT],
    )
    
    if RICH_AVAILABLE:
        console.print(f"Processing query: [bold cyan]{query}[/bold cyan]")
        console.print("[bold blue]Connecting to MLflow MCP server...[/bold blue]")
    else:
        print(f"Processing query: '{query}'")
        print("Connecting to MLflow MCP server...")
    
    try:
        # Connect to the MCP server
        async with stdio_client(server_params) as (read, write):
            logger.info("Connected to MLflow MCP server")
            if RICH_AVAILABLE:
                console.print("[bold green]Connected to server[/bold green]")
            else:
                print("Connected to server")
            
            async with ClientSession(read, write) as session:
                # Initialize the connection
                logger.info("Initializing session...")
                await session.initialize()
                logger.info("Session initialized successfully")
                
                # Get the available MCP tools
                logger.info("Loading MCP tools...")
                tools = await load_mcp_tools(session)
                logger.info(f"Successfully loaded {len(tools)} tools")
                
                # List tools for debugging
                for i, tool in enumerate(tools, 1):
                    logger.info(f"  Tool {i}: {tool.name}")
                
                # Create a ReAct agent
                logger.info("Creating agent...")
                agent = create_react_agent(model, tools)
                
                # Prepare messages with system prompt
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=query)
                ]
                
                # Process the query
                logger.info("Processing query through agent...")
                if RICH_AVAILABLE:
                    console.print("[bold yellow]Thinking...[/bold yellow]")
                else:
                    print("Thinking...")
                    
                agent_response = await agent.ainvoke({"messages": messages})
                
                logger.info("Query processing complete")
                return agent_response
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}


def display_response(response: Dict[str, Any]) -> None:
    """
    Display the agent's response in a user-friendly format.
    
    Args:
        response: The agent's response dictionary
    """
    if "error" in response:
        if RICH_AVAILABLE:
            console.print(f"\n[bold red]Error:[/bold red] {response['error']}")
        else:
            print(f"\nError: {response['error']}")
        return
    
    if "messages" not in response:
        if RICH_AVAILABLE:
            console.print("\n[bold red]Error:[/bold red] Unexpected response format")
        else:
            print("\nError: Unexpected response format")
        return
    
    # Find the last AI message with content
    ai_message = None
    for message in reversed(response["messages"]):
        if isinstance(message, AIMessage) and message.content:
            ai_message = message
            break
    
    if not ai_message:
        if RICH_AVAILABLE:
            console.print("\n[bold red]Error:[/bold red] No AI response found")
        else:
            print("\nError: No AI response found")
        return
    
    # Display the response
    if RICH_AVAILABLE:
        console.print("\n[bold green]MLflow Assistant:[/bold green]")
        markdown = Markdown(ai_message.content)
        console.print(markdown)
    else:
        print("\nMLflow Assistant:")
        print("-" * 80)
        print(ai_message.content)
        print("-" * 80)


def main():
    """Main function to parse arguments and run the query processing."""
    parser = argparse.ArgumentParser(
        description="Query MLflow using natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python mlflow_client.py "List all models in my MLflow registry"
          python mlflow_client.py "List all experiments in MLflow registry"
          python mlflow_client.py "Give me details about iris-model"
          python mlflow_client.py "Give me system info"
        """)
    )
    
    # Add arguments
    parser.add_argument(
        "query", 
        type=str,
        help="The natural language query to process"
    )
    
    args = parser.parse_args()
    
    try:
        # Run the query processing with error handling
        result = asyncio.run(process_query(args.query))
        
        # Display the results
        if RICH_AVAILABLE:
            console.print("\n[bold]Results:[/bold]")
        else:
            print("\n=== Results ===\n")
        
        display_response(result)
    
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[bold]Operation cancelled by user[/bold]")
        else:
            print("\nOperation cancelled by user")
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"\n[bold red]Error processing query:[/bold red] {str(e)}")
        else:
            print(f"\nError processing query: {str(e)}")
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
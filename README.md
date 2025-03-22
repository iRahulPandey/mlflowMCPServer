# MLflow MCP Agent: Natural Language Interface for MLflow

This project provides a natural language interface to MLflow via the Model Context Protocol (MCP). It allows you to query your MLflow tracking server using plain English, making it easier to manage and explore your machine learning experiments and models.

## Overview

MLflow MCP Agent consists of two main components:

1. **MLflow MCP Server** (`mlflow_server.py`): Connects to your MLflow tracking server and exposes MLflow functionality through the Model Context Protocol (MCP).

2. **MLflow MCP Client** (`mlflow_client.py`): Provides a natural language interface to interact with the MLflow MCP Server using a conversational AI assistant.

## Features

- **Natural Language Queries**: Ask questions about your MLflow tracking server in plain English
- **Model Registry Exploration**: Get information about your registered models
- **Experiment Tracking**: List and explore your experiments and runs
- **System Information**: Get status and metadata about your MLflow environment

## Prerequisites

- Python 3.8+
- MLflow server running (default: `http://localhost:8080`)
- OpenAI API key for the LLM

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/iRahulPandey/mlflowAgent.git
   cd mlflowAgent
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install mcp[cli] langchain-mcp-adapters langchain-openai langgraph mlflow
   ```

4. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

5. (Optional) Configure the MLflow tracking server URI:
   ```bash
   export MLFLOW_TRACKING_URI=http://localhost:8080
   ```

## Usage

### Starting the MCP Server

First, start the MLflow MCP server:

```bash
python mlflow_server.py
```

The server connects to your MLflow tracking server and exposes MLflow functionality via MCP.

### Making Queries

Once the server is running, you can make natural language queries using the client:

```bash
python mlflow_client.py "What models do I have registered in MLflow?"
```

Example Queries:

- "Show me all registered models in MLflow"
- "List all my experiments"
- "Get details for the model named 'iris-classifier'"
- "What's the status of my MLflow server?"

## Configuration

You can customize the behavior using environment variables:

- `MLFLOW_TRACKING_URI`: URI of your MLflow tracking server (default: `http://localhost:8080`)
- `OPENAI_API_KEY`: Your OpenAI API key
- `MODEL_NAME`: The OpenAI model to use (default: `gpt-3.5-turbo-0125`)
- `MLFLOW_SERVER_SCRIPT`: Path to the MLflow MCP server script (default: `mlflow_server.py`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

## Architecture

### MLflow MCP Server (`mlflow_server.py`)

The server connects to your MLflow tracking server and exposes the following tools via MCP:

- `list_models`: Lists all registered models in the MLflow model registry
- `list_experiments`: Lists all experiments in the MLflow tracking server
- `get_model_details`: Gets detailed information about a specific registered model
- `get_system_info`: Gets information about the MLflow tracking server and system


## Limitations

- Currently only supports a subset of MLflow functionality
- The client requires internet access to use OpenAI models
- Error handling may be limited for complex MLflow operations

## Future Improvements

- Add support for MLflow model predictions
- Improve the natural language understanding for more complex queries
- Add visualization capabilities for metrics and parameters
- Support for more MLflow operations like run management and artifact handling

## License

[MIT License](LICENSE)

## Acknowledgments

- [Model Context Protocol (MCP)](https://github.com/anthropics/model-context-protocol): For the protocol specification
- [LangChain](https://github.com/langchain-ai/langchain): For the agent framework
- [MLflow](https://github.com/mlflow/mlflow): For the tracking and model registry functionality
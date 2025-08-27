# IntelliAgent: Multi-Agent System with SAP AI Core

A sophisticated multi-agent system that leverages SAP AI Core for intelligent task execution, featuring a worker agent for complex task handling and an evaluator agent for response validation.

## 🌟 Features

- **Worker Agent (GPT-4.1)**
  - Complex task execution
  - Web browsing capabilities
  - Maintains conversation context
  - Tool integration for enhanced functionality

- **Evaluator Agent (Mistral-Small-Instruct)**
  - Response validation
  - Structured feedback generation
  - Task completion assessment
  - Iterative improvement through feedback loops

- **LangSmith Integration**
  - Comprehensive tracing and monitoring
  - Performance analytics
  - Debug capabilities
  - Test case tracking

## 📋 Prerequisites

- SAP AI Core account and credentials
- Python 3.x
- Network access to SAP AI Core endpoints
- Environment variables configured in `.env` file

## 🛠️ Required Packages

```bash
pip install "langgraph[all]" langchain-community langchain-openai langsmith gradio
```

## 🔧 Configuration

Create a `.env` file with the following variables:

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

## 📚 Project Structure

- `main.ipynb`: Main implementation notebook
- `llm_client.py`: SAP AI Core client implementation
- `intelliagent_memory.db`: SQLite database for persistent memory
- `README.md`: Project documentation

## 🚀 Getting Started

1. Clone the repository
2. Install required packages
3. Configure environment variables
4. Run the Jupyter notebook

## 💻 Usage

The system can be used through:

1. **Jupyter Notebook Interface**
   - Run cells sequentially
   - Execute test cases
   - Monitor performance

2. **Gradio Web Interface**
   - Interactive chat interface
   - Success criteria specification
   - Real-time feedback

## 🧪 Test Cases

The system includes four comprehensive test cases:

1. **Basic Human Interaction**
   - Tests self-awareness
   - Basic response capabilities

2. **Web Research**
   - Tests tool usage
   - Information gathering capabilities

3. **Feedback Loop**
   - Tests iterative improvement
   - Response refinement

4. **Complex Multi-step Tasks**
   - Tests comprehensive capabilities
   - Multiple tool usage
   - Complex reasoning

## 📊 Monitoring

- LangSmith dashboard for performance monitoring
- Tracing for all agent interactions
- Test case specific analytics
- Tool usage tracking

## 🔒 Security

- Secure credential management through environment variables
- SAP AI Core security standards
- Controlled access to external tools

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## ⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SAP AI Core team
- LangChain community
- LangSmith team
- Gradio team

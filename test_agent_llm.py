import os
import sys
sys.path.insert(0, 'src')

os.environ['AGENT_LLM_PROVIDER'] = 'ollama'
os.environ['AGENT_LLM_MODEL'] = 'llama3.1:8b'
os.environ['OLLAMA_BASE_URL'] = 'http://localhost:11434'
os.environ['CREDIT_RISK_DATA_MODE'] = 'mock'

from agent import graph

tests = [
    ('get_dashboard_metrics', 'Show dashboard metrics.'),
    ('get_model_comparison', 'Compare XGBoost and LSTM evaluation results.'),
    ('get_application_details', 'What is AMT_CREDIT for application ID 1001?'),
    ('get_prediction', 'Get the prediction for application ID 1001.'),
    ('explain_prediction', 'Explain the prediction for application ID 1001.'),
    ('generate_report', 'Generate a credit risk report.'),
]

for tool_name, question in tests:
    print(f'\n=== Testing {tool_name} ===')
    print(f'Question: {question}')
    state = graph.run_agent(question)
    print(f'Status: {state.get("status")}')
    print(f'Selected tool: {state.get("selected_tool")}')
    print(f'Tool args: {state.get("tool_arguments")}')
    response = state.get("final_response", "")
    print(f'Response (first 300 chars): {response[:300]}...')
    print()
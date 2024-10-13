import os
import sys
import requests
import json
import openai
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from genAgent import buildExternalEndpoints, buildTools
from advRAG import llm_advRAG
from typing import List
from langchain_core.tools import tool
from langchain_ollama import ChatOllama



openAPIspec = buildTools.readOpenAPISpec()
createdTools = buildTools.openAPItoTools(openAPIspec)

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
).bind_tools(createdTools)

checker_llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

non_tools_llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

def executeFunction(toolName, functionParams):
        print(f"Executing function {toolName} with args {functionParams}")
        endpointURL, httpMethod, headers = buildExternalEndpoints.buildRestEndpoint(toolName, functionParams)
        print('Built the rest endpoint  ' + endpointURL)
        try:
            response = requests.get(endpointURL, headers=headers)

            if not response.content:
                return f"{toolName} executed successfully"
            else:
                response.raise_for_status()
                data = response.json()
                print("data:   ", data)
                return data

        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

def check_tool_selection(tool_calls, user_prompt):
    checker_prompt = f"""
        As a checker agent, determine if the tool calls made are appropriate for the user prompt.

        User prompt: {user_prompt}
        Tool calls made: {tool_calls}

        Did the agent use the correct tools to answer the user prompt? Answer 'Yes' or 'No'.
    """
    checker_response = checker_llm.invoke(checker_prompt)
    print("Checker agent response:")
    print(checker_response.content)
    print('\n')
    return True if 'Yes' in checker_response.content else False

def agent_workflow(user_prompt):
    result = llm.invoke(user_prompt)
    print(result)
    if result.tool_calls:
        print("Possible Tool Call:")
        print(result.tool_calls)
        print('\n')
        if check_tool_selection(result.tool_calls, user_prompt):
            print("Tool calls accepted.")
            tool_name, function_name = result.tool_calls[0].get("name"), result.tool_calls[0].get("args")
            first_chat_response = executeFunction(tool_name, function_name)

            print(first_chat_response)
            print("Final result:")
            print(result.content)
            print('\n')
            return
        else:
            # For Regular Chat Completion
            # print("Checker agent did not accept the tool calls, so no tool calls made. Using regular chat completion: ")
            # completion_result_stream = non_tools_llm.stream(user_prompt)
            # for chunk in completion_result_stream:
            #     print(chunk.content, end='', flush=True)
            # print('\n')
            # return

            # FOR RAG
            print("Checker agent did not accept the tool calls, so no tool calls made. Using RAG: ")
            rag_agent = llm_advRAG.RAG()
            result = rag_agent.perform_RAG(user_prompt)
            print(result)
            print('\n')
            return



if __name__ == "__main__":
    while True:
        user_prompt = input("Enter your prompt (or type 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            print("Goodbye!")
            break
        agent_workflow(user_prompt)

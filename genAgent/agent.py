import os
import requests
import json
import openai
from dotenv import load_dotenv
load_dotenv()

from genAgent import buildExternalEndpoints, buildTools

class GeneralAgent:
    def __init__(self, systemMessage: str = None) -> None:
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.messages = []
        if systemMessage:
            self.messages.append({"role": "system", "content": systemMessage})
        else:
            self.messages.append({"role": "system", "content": ""})

        openAPIspec = buildTools.readOpenAPISpec()
        createdTools = buildTools.openAPItoTools(openAPIspec)
        self.tools = createdTools

    def executeFunction(self, toolName, functionParams):
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

    def answerQuery(self, query: str) -> str:
        self.messages.append({'role': 'user', 'content': query})

        completionArgs1 = {
            "model": "gpt-4",
            "messages": self.messages,
            "functions": self.tools,
            "function_call": "auto"
        }

        try:
            firstChatCompletion = openai.ChatCompletion.create(**completionArgs1)
            choice = firstChatCompletion['choices'][0]
            responseDict = choice.get('message', {})

            if responseDict.get("function_call"):
                funcCall = responseDict["function_call"]
                self.messages.append(responseDict)
                toolName = funcCall['name']
                try:
                    functionParams = json.loads(funcCall.get('arguments', '{}'))
                except json.JSONDecodeError as e:
                    print(f"Error decoding function arguments: {e}")
                    functionParams = {}

                try:
                    funcResponse = self.executeFunction(toolName, functionParams)
                    self.messages.append({
                        "role": "function",
                        "name": toolName,
                        "content": json.dumps(funcResponse),
                    })
                except Exception as e:
                    print(f"Error executing function {toolName}: {e}")
                    funcResponse = {"error": str(e)}
                    self.messages.append({
                        "role": "function",
                        "name": toolName,
                        "content": json.dumps(funcResponse),
                    })

                completionArgs2 = {
                    "model": "gpt-4",
                    "messages": self.messages
                }
                try:
                    secondModelResponse = openai.ChatCompletion.create(**completionArgs2)
                    resultResponse = secondModelResponse['choices'][0]['message']['content']
                    self.messages.append({'role': 'assistant', 'content': resultResponse})
                except Exception as e:
                    print(f"Chat completion failed: {e}")
                    self.messages = self.messages[:-2]
                    self.messages.append({'role': 'assistant', 'content': "I'm sorry, but the second chat completion failed."})
            else:
                resultResponse = responseDict.get('content', '')
                self.messages.append({'role': 'assistant', 'content': resultResponse})
        except Exception as e:
            print(f" Chat completion failed: {e}")
            self.messages = self.messages[:-1]
            self.messages.append({'role': 'assistant', 'content': "I'm sorry, but the chat completion failed."})

        return resultResponse

def main(userInput):
    agent = GeneralAgent()
    return agent.answerQuery(userInput)

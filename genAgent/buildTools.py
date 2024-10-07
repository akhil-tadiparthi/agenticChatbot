import os
import json
import jsonref
from pprint import pp

def readOpenAPISpec():
    with open('/Users/akhiltadiparthi/Documents/GitHub/agenticChatbot/genAgent/openAPIsampleSpec.json', 'r') as f:
        openAPIspec = jsonref.loads(f.read())
    return openAPIspec

def openAPItoTools(openAPIspec):
    tools = []
    for path, methods in openAPIspec["paths"].items():
        for method, specRef in methods.items():
            spec = jsonref.replace_refs(specRef)
            summary = spec.get("summary")
            desc = spec.get("description")
            schema = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            params = spec.get("parameters", [])
            for param in params:
                if "schema" in param:
                    parameterName = param["name"]
                    schema["properties"][parameterName] = {
                        "type": param["schema"].get("type", "string"),
                        "description": param.get("description", "")
                    }
                    if param.get("required", False):
                        schema["required"].append(parameterName)

            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": summary,
                        "description": desc,
                        "parameters": schema
                    }
                }
            )
    return tools
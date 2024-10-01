import os
import json
import jsonref
from pprint import pp
import urllib.parse
import jsonref
import urllib
from urllib.parse import quote
from genAgent import buildTools
from dotenv import load_dotenv

load_dotenv()

def extractEndpointSummary(openAPIspec, summary):
    for path, methods in openAPIspec['paths'].items():
        for method, details in methods.items():
            if details.get('summary') == summary:
                return path
    return None


def buildRestEndpoint(summary, funcArgs):
    openAPIspec = buildTools.readOpenAPISpec()
    baseUrl = openAPIspec.get('servers', [{}])[0].get('url', '')
    relativePath = extractEndpointSummary(openAPIspec, summary)
    httpMethod = None
    queryParams = {}
    # If you require AUTH token to access your backend uncomment the following code
    # AUTH_TOKEN = os.getenv('AUTH_TOKEN')
    headers = {
                'Content-Type': 'application/json',
                # 'Authorization': f'Bearer {AUTH_TOKEN}'
            }

    pathInfo = openAPIspec["paths"].get(relativePath)
    if pathInfo:
        for method in pathInfo:
            if method.lower() == 'get':
                httpMethod = 'GET'

                for param in pathInfo[method].get('parameters', []):
                    if param['in'] == 'query':
                        queryParams[param['name']] = funcArgs.get(param['name'], '')

        endpointUrl = f"{baseUrl}{relativePath}"
        if queryParams:
            query_string = '&'.join(f"{key}={urllib.parse.quote(value)}" for key, value in queryParams.items() if value is not None)
            endpointUrl += f"?{query_string}"


        return endpointUrl, httpMethod, headers
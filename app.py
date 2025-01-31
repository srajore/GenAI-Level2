import boto3
import json

bedrock= boto3.client('bedrock')
bedrock_runtime=boto3.client('bedrock-runtime')

response = bedrock.list_foundation_models()

print("Available Models: ")

print(json.dumps(response, indent=2))


model_id = 'amazon.titan-text-express-v1'
prompt= "Describe the purpose of 'hello world' program in one line"

request_body = json.dumps({
    "inputText": prompt,
    "textGenerationConfig": {
        "maxTokenCount": 512,
        "temperature": 0.7,
        "topP": 0.9
    }
})


response1=bedrock_runtime.invoke_model(
    modelId=model_id,
    body=request_body,
    contentType='application/json',
    accept='application/json'
)

response_body =json.loads(response1['body'].read())['results'][0]['outputText']

print("\n Model Resonse: \n")

print(response_body)
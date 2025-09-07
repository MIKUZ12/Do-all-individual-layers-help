from openai import OpenAI
client = OpenAI(
    api_key='sk-123456',
    base_url="http://0.0.0.0:23333/v1"
)
model_name = client.models.list().data[0].id
print(model_name)
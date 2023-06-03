import requests

url = 'https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json'
file_path = 'deviation.json'

response = requests.get(url)
with open(file_path, 'wb') as file:
    file.write(response.content)

import requests
  
# defining the api-endpoint  
API_ENDPOINT = "http://anur0narm.pythonanywhere.com/"
  
data = {'action': 'runner_script'}
  
# sending post request and saving response as response object 
r = requests.post(url = API_ENDPOINT, json=data, timeout=100)

print('Site hit:\n'+str(r.text))
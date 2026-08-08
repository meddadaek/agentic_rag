import requests

url = 'https://en.wikipedia.org/wiki/India'
try:
    r = requests.get(url, timeout=10)
    print('Status:', r.status_code)
    print('Content-Type:', r.headers.get('content-type'))
    print('Body (first 300 chars):')
    print(r.text[:300])
except Exception as e:
    print('ERROR HTTP:', repr(e))

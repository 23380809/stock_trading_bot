import requests

class RequestAPI():
    def request(self, method, url, header=None, **kwargs):
        if not header:
            headers = {"Content-type": "application/json; charset=utf-8"}
        resp = requests.request(method, url, headers=headers, **kwargs)
        return resp

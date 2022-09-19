from RequestAPI import RequestAPI
import json

class crypto_api():
    def __init__(self):
        self.init = RequestAPI()
    
    def get_price(self, id, currency):
        url = "https://api.coingecko.com/api/v3/simple/price?ids={}&vs_currencies={}".format(id, currency)
        resp = self.init.request("get", url)
        return resp


# test = crypto_api()
# print(test.get_price('bitcoin', 'usd').text)
# data = json.loads(test.get_price('bitcoin', 'usd').text)


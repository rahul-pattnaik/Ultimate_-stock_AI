import requests

def get_nifty500():

    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    r = requests.get(url, headers=headers)

    data = r.json()

    stocks = [x["symbol"] + ".NS" for x in data["data"]]

    return stocks
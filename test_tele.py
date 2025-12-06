import requests

TOKEN = "8137546763:AAEOr8zLInmdnL-lCluCoSxJKBLEOlyj-G0"  # Token của bạn
CHAT_ID = "6448037928"  # Chat ID của bạn

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
data = {
    "chat_id": CHAT_ID,
    "text": "🧪 Test message từ SafeZone Bot!"
}

response = requests.post(url, data=data)
print(response.json())




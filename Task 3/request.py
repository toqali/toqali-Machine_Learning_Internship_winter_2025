import requests

url = 'http://localhost:5000/predict'
data = {
    "number_of_adults": "",
    "number_of_children": "",
    "Number of Weekend Nights": "",
    "Number of Week Nights": "",
    "type_of_meal": "",
    "room_type": "",
    "market_segment_type": "",
    "lead_time": "",
    "average_price": "",
    "special_requests": ""  ,
    "reservation_day": "",
    "reservation_month": "",
    "reservation_year": ""
}

response = requests.post(url, data=data)
print(response.text)

if response.status_code == 200:  # Check if the request was successful
    print("Request successful:", response.text)
else:
    print("Request failed with status code:", response.status_code)
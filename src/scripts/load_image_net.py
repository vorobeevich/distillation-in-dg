import requests
from urllib.parse import urlencode

base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = 'https://disk.yandex.ru/d/mZA7LkzPhY6f9g'  # your link

# download link
final_url = base_url + urlencode(dict(public_key=public_key))
response = requests.get(final_url)
download_url = response.json()["href"]

# load and save file
download_response = requests.get(download_url)
with open("samples.zip", "wb") as f:   # Здесь укажите нужный путь к файлу
    f.write(download_response.content)
import requests
from bs4 import BeautifulSoup

# URL của trang chứa danh sách bệnh
url = "https://youmed.vn/tin-tuc/trieu-chung-benh/"

# Gửi yêu cầu GET để lấy nội dung trang
response = requests.get(url)

# Kiểm tra xem yêu cầu có thành công không
if response.status_code == 200:
    # Phân tích cú pháp HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Tìm tất cả các thẻ <a> có trong trang
    links = soup.find_all('a', href=True)

    # In ra tất cả các URL chứa trong thuộc tính href
    for link in links:
        href = link['href']
        if 'youmed.vn/tin-tuc' in href:
            print(href)
else:
    print(f"Không thể tải trang, mã lỗi: {response.status_code}")


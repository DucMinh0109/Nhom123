import requests
from bs4 import BeautifulSoup

url = "https://youmed.vn/tin-tuc/trieu-chung-benh/"
victim1 = "https://youmed.vn/tin-tuc/benh-addison-nguyen-nhan-trieu-chung-va-dieu-tri/"
victim2 = "https://youmed.vn/tin-tuc/o-nong-nguyen-nhan-dieu-tri-cach-phong-ngua/"
# Gửi yêu cầu GET để lấy nội dung trang
response = requests.get(url)

# Kiểm tra xem yêu cầu có thành công không
if response.status_code == 200:
    # Phân tích cú pháp HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Tìm tất cả các thẻ <a> có trong trang
    links = soup.find_all('a', href=True)

    # Biến kiểm soát trạng thái có phải là trang bắt đầu không
    is_collecting = False

    # In ra tất cả các URL chứa trong thuộc tính href
    for link in links:
        href = link['href']

        # In liên kết nếu nó chứa 'youmed.vn/tin-tuc'
        if 'youmed.vn/tin-tuc' in href:
            if href == victim1:
                is_collecting = True
            if is_collecting:
                print(f'"{href}",')
            if href == victim2:
                print(f'"{href}",')
                break

        # In ra liên kết nếu đang trong phạm vi thu thập


        # Dừng thu thập khi gặp end_url


else:
    print(f"Không thể tải trang, mã lỗi: {response.status_code}")

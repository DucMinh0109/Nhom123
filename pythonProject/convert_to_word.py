import csv
from docx import Document

# Hàm đọc dữ liệu từ file DOCX
def read_data_from_word(file_path):
    doc = Document(file_path)
    data = []
    current_disease = {}

    # Duyệt qua tất cả các đoạn văn trong file DOCX
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text.startswith("Tên bệnh:"):
            if current_disease:
                data.append(current_disease)
            current_disease = {"Disease": text.replace("Tên bệnh:", "").strip()}
        elif text.startswith("Triệu chứng:"):
            current_disease["Symptoms"] = parse_multiline_data(text, "Triệu chứng:")
        elif text.startswith("Nguyên nhân phỏng đoán:"):
            current_disease["Cause"] = text.replace("Nguyên nhân phỏng đoán:", "").strip()
        elif text.startswith("Phòng ngừa:"):
            current_disease["Prevention"] = parse_multiline_data(text, "Phòng ngừa:")
        elif text.startswith("Điều trị:"):
            current_disease["Treatment"] = text.replace("Điều trị:", "").strip()

    # Thêm bệnh cuối cùng vào danh sách
    if current_disease:
        data.append(current_disease)

    return data

# Hàm phân tách các dữ liệu nhiều dòng thành danh sách
def parse_multiline_data(text, prefix):
    data = text.replace(prefix, "").strip()
    if data:
        return [item.strip() for item in data.split(",") if item.strip()]
    return []

# Hàm chuyển đổi dữ liệu từ DOCX sang CSV
def write_to_csv(disease_data, output_file):
    # Mở file CSV để ghi
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Disease", "Symptoms", "Cause", "Prevention", "Treatment"])
        writer.writeheader()

        # Ghi từng dòng dữ liệu bệnh vào file CSV
        for disease in disease_data:
            writer.writerow(disease)

# Đọc dữ liệu từ file DOCX
file_path = "disease_data.docx"
disease_data = read_data_from_word(file_path)

# Ghi dữ liệu vào file CSV
output_file = "disease_data.csv"
write_to_csv(disease_data, output_file)

print(f"Dữ liệu đã được chuyển thành công vào file {output_file}")

from flask import Flask, render_template, request, jsonify
from docx import Document
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Initialize Flask app
app = Flask(__name__)


# Load the disease data from the Word document
def read_data_from_word(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File không tồn tại: {file_path}")

        doc = Document(file_path)
        data = []
        current_disease = {}

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text.startswith("Tên bệnh:"):
                if current_disease:
                    data.append(current_disease)
                current_disease = {"Disease": text.replace("Tên bệnh:", "").strip()}
            elif text.startswith("Triệu chứng:"):
                current_disease["Symptoms"] = text.replace("Triệu chứng:", "").strip()
            elif text.startswith("Nguyên nhân phỏng đoán:"):
                current_disease["Cause"] = text.replace("Nguyên nhân phỏng đoán:", "").strip()
            elif text.startswith("Phòng ngừa:"):
                current_disease["Prevention"] = text.replace("Phòng ngừa:", "").strip()
            elif text.startswith("Điều trị:"):
                current_disease["Treatment"] = text.replace("Điều trị:", "").strip()

        if current_disease:
            data.append(current_disease)
        return data
    except Exception as e:
        print(f"Error reading Word file: {e}")
        return []


# Chuẩn bị dữ liệu huấn luyện
def prepare_data(disease_data):
    # Chuyển triệu chứng thành vector
    vectorizer = CountVectorizer()
    symptoms_list = [disease["Symptoms"] for disease in disease_data]
    X = vectorizer.fit_transform(symptoms_list).toarray()  # Vector hóa triệu chứng
    y = [disease["Disease"] for disease in disease_data]  # Nhãn là tên bệnh
    return X, y, vectorizer


# Hàm tính phần trăm xác suất dựa trên triệu chứng khớp
def calculate_probability(input_symptoms, disease_symptoms):
    matched_symptoms = [sym for sym in input_symptoms if sym in disease_symptoms]
    matched_count = len(matched_symptoms)
    total_symptoms = len(disease_symptoms)

    # Cộng 10% cho 5 triệu chứng đầu tiên
    probability = 0
    for i in range(min(matched_count, 5)):
        probability += 10

    # Cộng phần còn lại với 80% chia đều cho các triệu chứng
    if matched_count > 5:
        remaining_probability = 80  # Đã cộng 50% cho 5 triệu chứng đầu tiên
        additional_probability_per_symptom = remaining_probability / total_symptoms
        for i in range(5, matched_count):
            probability += additional_probability_per_symptom

    # Đảm bảo tổng xác suất không vượt quá 80%
    return min(probability, 80)


# Hàm tính khoảng cách Euclidean
def calculate_knn_distance(input_symptom, disease_symptom, vectorizer, disease):
    # Chuyển triệu chứng thành vector (sử dụng CountVectorizer đã huấn luyện)
    input_vector = vectorizer.transform([input_symptom]).toarray()
    disease_vector = vectorizer.transform([disease_symptom]).toarray()

    # Tính khoảng cách Euclidean giữa 2 vector triệu chứng
    distance = euclidean_distances(input_vector, disease_vector)[0][0]

    # In ra khoảng cách giữa triệu chứng người dùng và triệu chứng bệnh
    print(f"Khoảng cách giữa triệu chứng nhập vào và triệu chứng bệnh: {distance}, {disease}")
    return distance


def search_diseases_by_symptom(input_symptom):
    input_symptoms = [sym.strip().lower() for sym in input_symptom.split(",")]

    results = []

    for disease in disease_data:
        disease_symptoms = [sym.strip().lower() for sym in disease["Symptoms"].split(",")]

        # Tính xác suất của triệu chứng khớp
        probability = calculate_probability(input_symptoms, disease_symptoms)

        # Dự đoán bệnh bằng KNN (sử dụng khoảng cách Euclidean thay vì KNN trực tiếp)
        distance = calculate_knn_distance(input_symptom, disease["Symptoms"], vectorizer, disease)

        # Thêm kết quả vào danh sách nếu xác suất > 0
        if probability > 0:
            results.append({
                "Disease": disease["Disease"],
                "Probability": round(probability, 2),  # Làm tròn đến 2 chữ số thập phân
                "Matched Symptoms": ", ".join([sym for sym in input_symptoms if sym in disease_symptoms]),
                "Symptoms": disease["Symptoms"],
                "Cause": disease.get("Cause", "Không có thông tin"),
                "Prevention": disease.get("Prevention", "Không có thông tin"),
                "Treatment": disease.get("Treatment", "Không có thông tin"),
                "Distance": round(distance, 2),  # Khoảng cách ban đầu
            })

    # Trừ đi khoảng cách của từng phần tử trong results / 1000
    for result in results:
        result["Probability"] -= result["Distance"] / 100  # Trừ đi 1 phần 1000 của khoảng cách

    # Sắp xếp kết quả theo xác suất giảm dần. Nếu xác suất bằng nhau, sắp xếp theo khoảng cách tăng dần
    results.sort(key=lambda x: (-x["Probability"], x["Distance"]))

    return results


@app.route("/")
def home():
    return render_template("index.html", diseases=disease_data)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        symptoms = request.form.get("symptoms", "").lower().strip()
        if not symptoms:
            return render_template("index.html", message="Vui lòng nhập triệu chứng.")

        # Tìm kiếm bệnh dựa trên triệu chứng nhập vào
        results = search_diseases_by_symptom(symptoms)

        return render_template("index.html", diseases=disease_data, results=results, input_symptoms=symptoms)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template("index.html", message="Đã xảy ra lỗi, vui lòng thử lại.")


if __name__ == "__main__":
    # Load dữ liệu bệnh
    disease_data = read_data_from_word("disease_data.docx")
    # Chuẩn bị dữ liệu huấn luyện
    X_train, y_train, vectorizer = prepare_data(disease_data)

    app.run(debug=True)

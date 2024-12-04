import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Khởi tạo ứng dụng Dash
app = dash.Dash(__name__)

# Dữ liệu mẫu
data = {
    'Disease': ['Cảm cúm', 'Viêm phổi', 'Sốt xuất huyết', 'COVID-19'],
    'Probability': [75, 50, 65, 90],
}

df = pd.DataFrame(data)

# Biểu đồ cột
fig = px.bar(df, x='Disease', y='Probability', title="Xác suất bệnh lý dựa trên triệu chứng")

# Cấu trúc giao diện
app.layout = html.Div([
    html.H1("Hệ thống Dự đoán Bệnh lý", style={'text-align': 'center'}),
    dcc.Graph(figure=fig),
])

# Chạy ứng dụng
if __name__ == '__main__':
    app.run_server(debug=True)

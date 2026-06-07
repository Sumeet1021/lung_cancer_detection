from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_pdf(filename, patient_data):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, "Lung Cancer Prediction Report")

    y = 700
    for key, value in patient_data.items():
        c.drawString(100, y, f"{key}: {value}")
        y -= 20

    c.save()
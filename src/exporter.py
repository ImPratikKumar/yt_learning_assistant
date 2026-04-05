import os
import io
from typing import Literal
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_pdf(title, content, content_type: Literal["subtitle", "summary"]):

    # 1. Create an in-memory buffer
    buffer = io.BytesIO()

    # 2. Pass the buffer (not a filename) to SimpleDocTemplate
    # pdf = SimpleDocTemplate(buffer)

    dir = "E:\Projects\youtube_learning_assistant\data"

    if content_type == "subtitle":
        file_path = os.path.join(dir, f"subtitle_{title}.pdf")
    elif content_type == "summary":
        file_path = os.path.join(dir, f"summary_{title}.pdf")
    

    pdf = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    if content_type == "subtitle":
        pdf_title = f"YouTube video id: {title} subtitle"
    elif content_type == "summary":
        pdf_title = f"YouTube video id: {title} summary"

    custom_title_style = ParagraphStyle(
        name="CustomTItleStyle",
        fontSize=20,
        leading=24,
        alignment=1, # 1 -> center
        spaceAfter=20
    )

    pdf_content = []
    pdf_content.append(Paragraph(pdf_title, custom_title_style))
    pdf_content.append(Paragraph(content, styles["Normal"]))
    
    # 3. Build the PDF into the buffer
    pdf.build(pdf_content)

    # 4. Get the bytes from the buffer and return then
    # pdf_bytes = buffer.getvalue()
    # buffer.close()
    # return pdf_bytes
















# from fpdf import FPDF

# def generate_pdf(title, content):
#     pdf = FPDF()
#     pdf.add_page()

#     pdf.add_font("DejaVu", style="", fname="DejaVuSans.ttf")
#     # Add a Title
#     pdf.set_font("DejaVu", 'B', 16)
#     pdf.cell(0, 10, title.encode('latin-1', 'replace').decode('latin-1'), ln=True, align='C')
#     pdf.ln(10) ## Line break

#     # Add the Content
#     pdf.set_font("DejaVu", size=12)
#     # multo_cell handles text wrapping automatically
#     pdf.multi_cell(0, 10, content)

#     # Return the PDF as a byte string for streamlit
#     return pdf.output(dest='S')
import os
from django.http import HttpResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image
from PIL.ExifTags import TAGS
from django.shortcuts import render
from django.core.files.storage import default_storage

def base_file(request):
    return render(request, "verifier/base.html")  # Looks inside templates/verifier/templates/

def extract_metadata(file_path):
    """Extract metadata from image files."""
    try:
        img = Image.open(file_path)
        meta_info = img._getexif()

        extracted_meta = {}
        if meta_info:
            for tag, value in meta_info.items():
                tag_name = TAGS.get(tag, tag)
                extracted_meta[tag_name] = value

        return extracted_meta
    except Exception as e:
        return {"Error": str(e)}

def report_file(request):
    if request.method == "POST" and request.FILES.get("document"):
        uploaded_file = request.FILES["document"]

        # Save the uploaded file temporarily
        file_path = default_storage.save(f"temp/{uploaded_file.name}", uploaded_file)
        abs_file_path = default_storage.path(file_path)

        # Extract metadata
        meta_data = extract_metadata(abs_file_path)

        # Create the PDF response
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="meta_report.pdf"'

        # Create the PDF object
        p = canvas.Canvas(response, pagesize=letter)
        width, height = letter

        # Title
        p.setFont("Helvetica-Bold", 16)
        p.drawString(200, height - 50, "Document Meta Report")

        # Meta Data Content
        y_position = height - 100
        p.setFont("Helvetica", 12)
        
        for key, value in meta_data.items():
            p.drawString(100, y_position, f"{key}: {value}")
            y_position -= 20

        # Save the PDF
        p.showPage()
        p.save()

        # Cleanup (delete the temporary file)
        default_storage.delete(file_path)

        return response

    return render(request, "verifier/upload.html")

def upload_file(request):
    return render(request, "verifier/upload.html")
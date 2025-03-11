import os
from django.http import HttpResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from PIL import Image
from PIL.ExifTags import TAGS
from django.shortcuts import render
from django.core.files.storage import default_storage

def base_file(request):
    return render(request, "verifier/base.html")  # Looks inside templates/verifier/

def register_file(request):
    return render(request,"verifier/register.html")

def login_file(request):
    return render(request,"verifier/login.html")

def upload_file(request):
    return render(request, "verifier/upload.html")

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
        print(extracted_meta)
        return extracted_meta
    except Exception as e:
        print(e)
        return {"Error": str(e)}
    
from PIL import Image, ImageChops, ImageEnhance
from io import BytesIO

def ela(image_path):
    try:
        # Open the original image
        original = Image.open(image_path).convert("RGB")

        # Save the image at lower quality for ELA analysis
        temp_path = image_path + "_temp.jpg"
        original.save(temp_path, quality=90)

        # Open the recompressed image
        temp_image = Image.open(temp_path)

        # Compute difference (ELA)
        diff = ImageChops.difference(original, temp_image)

        # Enhance brightness to highlight changes
        enhancer = ImageEnhance.Brightness(diff)
        diff = enhancer.enhance(10)  # Increase brightness

        # Save ELA image in memory as a byte stream
        ela_image_stream = BytesIO()
        diff.save(ela_image_stream, format="JPEG")
        ela_image_stream.seek(0)  # Reset stream position to the beginning

        # Save the byte stream as a temporary file
        ela_image_path = default_storage.save("temp/ela_output.jpg", ela_image_stream)
        ela_image_abs_path = default_storage.path(ela_image_path)

        # Clean up temporary image
        os.remove(temp_path)

        # Return the file path of the saved ELA image
        return ela_image_abs_path

    except Exception as e:
        return None


def report_file(request):
    if request.method == "POST" and request.FILES.get("document"):
        uploaded_file = request.FILES["document"]

        # Save the uploaded file temporarily
        file_path = default_storage.save(f"temp/{uploaded_file.name}", uploaded_file)
        abs_file_path = default_storage.path(file_path)

        # Extract metadata
        meta_data = extract_metadata(abs_file_path)

        #ELA Image
        ela_image_path = ela(abs_file_path)
        # Create a PDF response
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="meta_report.pdf"'

        p = canvas.Canvas(response, pagesize=letter)
        width, height = letter
        y_position = height - 50  # Initial position for writing text

        # 🟢 1. **Document Report Headline**
        p.setFont("Helvetica-Bold", 18)
        p.drawString(200, y_position, "📄 Document Verification Report")
        y_position -= 30  # Move down

        # 🟢 2. **User's Uploaded Image**
        try:
            img = Image.open(abs_file_path)
            img.thumbnail((250, 250))  # Resize image to fit PDF
            img_path = abs_file_path + "_thumb.jpg"
            img.save(img_path)
            p.drawImage(ImageReader(img_path), 100, y_position - 250, width=200, height=200)
            y_position -= 270  # Move below image
            os.remove(img_path)  # Delete temp thumbnail
        except Exception as e:
            p.drawString(100, y_position, "❌ Error displaying image in report.")
            y_position -= 20

        # 🟢 3. **Pass 1: Metadata Analysis**
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, y_position, "🔍 Pass 1: Metadata Analysis")
        y_position -= 20

        if not meta_data:  # If metadata is empty
            p.setFont("Helvetica", 12)
            p.drawString(100, y_position, "⚠️ No metadata found. Possible metadata stripping.")
            y_position -= 20
        else:
            p.setFont("Helvetica", 12)
            for key, value in meta_data.items():
                p.drawString(100, y_position, f"{key}: {value}")
                y_position -= 20

        # Pass 2: ELA Analysis (using saved ELA image)
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, y_position, "🔍 Pass 2: ELA Analysis")
        y_position -= 20

        try:
            p.drawImage(ImageReader(ela_image_path), 100, y_position - 250, width=200, height=200)
            y_position -= 250
        except Exception as e:
            p.drawString(100, y_position, "❌ Error displaying ELA image in report.")
            y_position -= 20


        # 🟢 4. **Result of Pass 1**
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, y_position, "📌  Result:")
        y_position -= 20

        # Check for suspicious metadata
        sus_reasons = is_suspicious(meta_data)

        if sus_reasons:
            p.setFont("Helvetica", 12)
            p.drawString(100, y_position, "❌ Suspicious Metadata Detected:")
            y_position -= 20
            for reason in sus_reasons:
                p.drawString(120, y_position, f"- {reason}")
                y_position -= 20
        else:
            p.setFont("Helvetica", 12)
            p.drawString(100, y_position, "✅ Pass 1 Passed: No suspicious metadata found.")

        # Save the PDF
        p.showPage()
        p.save()

        # Cleanup temporary files
        default_storage.delete(file_path)

        return response

    return render(request, "verifier/upload.html")

def is_suspicious(meta_data):
    """Checks if metadata is suspicious based on predefined rules."""
    sus_reasons = []

    

    if "Software" in meta_data and meta_data["Software"] not in ["Adobe Acrobat", "Microsoft Word"]:
        sus_reasons.append(f"File edited with suspicious software: {meta_data['Software']}")

    if "DateTimeOriginal" in meta_data and "DateTime" in meta_data:
        if meta_data["DateTimeOriginal"] > meta_data["DateTime"]:
            sus_reasons.append("Modification date is before creation date (Possible tampering).")

    if "GPSInfo" in meta_data:
        sus_reasons.append("GPS data detected (Check if location is relevant).")

    return sus_reasons


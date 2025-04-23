from PIL import Image
import os
import tempfile
import os
from datetime import datetime
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.platypus.flowables import HRFlowable
from PIL import Image, ImageChops, ImageEnhance, ExifTags
import piexif
import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import kagglehub
from .models import Profile
from django.http import HttpResponse
from io import BytesIO
import tempfile
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus.flowables import HRFlowable
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from django.http import FileResponse

# Load the pre-trained model
model = load(r'C:\Users\Anushka\Desktop\Django projects\DocumentVerify\savedModels\train_models.joblib')

def home(request):
    return render(request, "verifier/home.html")  # Looks inside templates/verifier/

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        if password == confirm_password:
            if not User.objects.filter(username=username).exists():
                User.objects.create_user(username=username, email=email, password=password)
                messages.success(request, "Registration successful!")
                return redirect('login')
            else:
                messages.error(request, "Username already exists!")
        else:
            messages.error(request, "Passwords do not match!")
    return render(request,"verifier/register.html")

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, "Login successful!")
            return redirect('home')  # Redirect to upload.html
        else:
            messages.error(request, "Invalid username or password!")

    return render(request, "verifier/login.html")

def verify(request):
    return render(request, "verifier/verify.html")

def upload(request):
    # sepal_length = request.GET['sepal_length']
    # sepal_width = request.GET['sepal_width']
    # petal_length = request.GET['petal_length']
    # petal_width = request.GET['petal_width']

    # # Make a prediction using the model
    # y_pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

    # # Convert numerical prediction to class name
    # if y_pred[0] == 0:
    #     y_pred = 'Setosa'
    # elif y_pred[0] == 1:
    #     y_pred = 'Versicolor'
    # else:
    #     y_pred = 'Virginica'

    # Render the result in the 'result.html' template
    return render(request, 'verifier/upload.html', {'result': y_pred})


def profile(request):
    return render(request,"verifier/profile.html")


@login_required
def upload_profile_picture(request):
    if request.method == "POST" and request.FILES.get("profile_picture"):
        request.user.profile.profile_picture = request.FILES["profile_picture"]
        request.user.profile.save()
    return redirect("profile")

@login_required
def delete_profile_picture(request):
    request.user.profile.profile_picture.delete()
    return redirect("profile")

@login_required
def upload_background(request):
    if request.method == "POST" and request.FILES.get("background_image"):
        request.user.profile.background_image = request.FILES["background_image"]
        request.user.profile.save()
    return redirect("profile")

@login_required
def delete_background(request):
    request.user.profile.background_image.delete()
    return redirect("profile")

def extract_ela(image_path):
    """Extract Error Level Analysis (ELA) feature."""
    try:
        image = Image.open(image_path).convert("RGB")
        image.save("temp.jpg", "JPEG", quality=90)
        temp_image = Image.open("temp.jpg")
        ela_image = ImageChops.difference(image, temp_image)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        return max_diff
    except:
        return None

def extract_metadata(image_path):
    """Extract metadata from image."""
    try:
        exif_data = piexif.load(image_path)
        date_time = exif_data["0th"].get(piexif.ImageIFD.DateTime, b'').decode()
        return 1 if date_time else 0 
    except:
        return 0

def extract_features(image_path):
    """Extract all features from an image."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_resized = cv2.resize(img, (224, 224))  # Resize for consistency

    # Channel-wise statistics
    mean_r, mean_g, mean_b = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
    std_r, std_g, std_b = np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2])

    # Extract ELA and Metadata
    ela_value = float(extract_ela(image_path) or 0)
    metadata_value = float(extract_metadata(image_path) or 0)

    # Flatten first 1024 pixels from resized image
    img_flatten = img_resized.flatten()[:1024]

    # Combine features
    features = np.hstack([
        mean_r, std_r,
        mean_g, std_g,
        mean_b, std_b,
        ela_value, metadata_value,
        img_flatten
    ])

    return features.astype(np.float32)

def apply_ela(image_path, quality=90):
    """Apply Error Level Analysis (ELA) to detect forgery."""
    original = Image.open(image_path).convert('RGB')
    temp_path = "temp_compressed.jpg"
    original.save(temp_path, 'JPEG', quality=quality)
    compressed = Image.open(temp_path)

    ela_image = ImageChops.difference(original, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    ela_path = "ela_output.png"
    ela_image.save(ela_path)
    return ela_path

def extract_metadata_for_report(image_path):
    """Extract metadata information from the image (if available)."""
    try:
        exif_data = Image.open(image_path)._getexif()
        if exif_data:
            metadata = {
                key: exif_data[key] for key in exif_data if key in [306, 271, 272]  # DateTime, Camera Make & Model
            }
            return metadata if metadata else 0  # Return 0 instead of "No Metadata Found"
        return 0  # If no metadata is found, return numerical 0
    except:
        return 0  # Ensure numeric return value

def report_file(request):
    if request.method == 'POST':
        if 'document' not in request.FILES:
            return HttpResponse("No file uploaded!", status=400)

        uploaded_file = request.FILES['document']
        try:
            # Verify the uploaded file
            try:
                img = Image.open(uploaded_file)
                img.verify()  # Verify if the file is a valid image
            except Exception:
                return HttpResponse("Uploaded file is not a valid image.", status=400)

            # Convert to .jpg if necessary
            temp_path = default_storage.path('temp_upload.jpg')
            img = Image.open(uploaded_file)
            img.convert('RGB').save(temp_path, 'JPEG')  # Convert to RGB and save as .jpg

            # Extract features
            features = extract_features(temp_path)
            if features is None:
                return HttpResponse("Invalid file or unable to process the image.", status=400)

            # Reshape for single prediction
            features = features.reshape(1, -1)

            feature_names = [
            "mean_r", "std_r", "mean_g", "std_g", "mean_b", "std_b",
            "ela_value", "metadata_value"] + [f"feat_{i}" for i in range(1024)]  # if you added 1024 pixel values

# Create a DataFrame with correct column names
            features_df = pd.DataFrame(features, columns=feature_names)

# Predict using the DataFrame
            prediction = model.predict(features_df)
            proba = model.predict_proba(features_df)
            
            print("Prediction:", prediction)
            # Return results
            is_fake = proba[0][1] > 0.4
            result = 'Fake' if is_fake else 'Real'  # This should be a string, not a tuple

                
            print(proba,result)
            report_path = generate_verification_report(temp_path, proba[0], result)
            return FileResponse(open(report_path, 'rb'), as_attachment=True, filename='Document_Verification_Report.pdf')

        except Exception as e:
            return HttpResponse(f"Error: {str(e)}", status=500)

        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_verification_report(image_path, prediction_proba, result):
    report_name = "document_verification_report.pdf"
    doc = SimpleDocTemplate(report_name, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)

    is_fake = "fake" in result.lower()
    status = "FORGED" if is_fake else "GENUINE"

    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    normal_style = styles['Normal']
    section_style = ParagraphStyle('Section', parent=styles['Heading2'], spaceAfter=12, textColor=colors.darkblue)

    elements = []
    elements.append(Paragraph("Document Verification Report", title_style))
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("Document Details", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.darkblue, spaceAfter=10))

    data = [
        ["Document Name:", os.path.basename(image_path)],
        ["Verification Date:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ["Forgery Status:", status],
        ["Detection Result:", result]
    ]

    elements.append(Spacer(1, 0.15 * inch))
    elements.append(Paragraph("Original Document", section_style))

    img = Image.open(image_path)
    img_width, img_height = img.size
    aspect_ratio = img_height / img_width
    img_width = 5 * inch
    img_height = img_width * aspect_ratio
    img_path = "original_resized.jpg"
    img.save(img_path)
    elements.append(RLImage(img_path, width=img_width, height=img_height))
    elements.append(Spacer(1, 0.25 * inch))

    detail_table = Table(data, colWidths=[2 * inch, 3.5 * inch])
    detail_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(detail_table)
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("Prediction Results", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.darkblue, spaceAfter=10))

    plt.figure(figsize=(5, 5))
    labels = ["Real", "Fake"]
    plt.pie(prediction_proba, labels=labels, autopct='%1.1f%%',
            colors=['#4ECDC4', '#FF6B6B'], startangle=140,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    plt.title("Forgery Prediction Confidence")
    pie_chart_path = "prediction_pie_chart.png"
    plt.savefig(pie_chart_path, bbox_inches='tight', dpi=150)
    plt.close()

    elements.append(Paragraph("Note: A document is considered fake if the Real probability is below 40%.", normal_style))
    elements.append(Spacer(1, 0.15 * inch))
    elements.append(RLImage(pie_chart_path, width=4 * inch, height=4 * inch))
    elements.append(Spacer(1, 0.15 * inch))

    prob_data = [["Prediction", "Confidence"],
                 ["Real", f"{prediction_proba[0]:.2%}"],
                 ["Fake", f"{prediction_proba[1]:.2%}"]]

    prob_table = Table(prob_data, colWidths=[2 * inch, 2 * inch])
    prob_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(prob_table)
    elements.append(Spacer(1, 0.25*inch))

     # Error Level Analysis
    elements.append(Paragraph("Error Level Analysis (ELA)", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.darkblue, spaceAfter=10))
    elements.append(Paragraph("ELA highlights differences in compression levels. Areas with higher error levels may indicate manipulation.", normal_style))
    elements.append(Spacer(1, 0.15*inch))

    # Apply ELA and add to report
    ela_path = apply_ela(image_path)
    elements.append(RLImage(ela_path, width=5*inch, height=3*inch))
    elements.append(Spacer(1, 0.25*inch))

    # Extract Metadata
    elements.append(Paragraph("Metadata Information", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.darkblue, spaceAfter=10))

    metadata = extract_metadata_for_report(image_path)
    if isinstance(metadata, dict) and metadata:
        metadata_rows = []
        for key, value in metadata.items():
            metadata_rows.append([str(key), str(value)])

        metadata_table = Table(metadata_rows, colWidths=[2*inch, 3.5*inch])
        metadata_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(metadata_table)
    else:
        elements.append(Paragraph("No metadata found in the image.", normal_style))

    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("Conclusion", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.darkblue, spaceAfter=10))

    conclusion_text = f"Based on our analysis, this document appears to be {status}."
    elements.append(Paragraph(conclusion_text, normal_style))

    if is_fake:
        elements.append(Paragraph("Signs of manipulation detected. Please refer to the ELA and chart.", normal_style))
    else:
        elements.append(Paragraph("No significant signs of manipulation detected.", normal_style))

    elements.append(Spacer(1, 0.25 * inch))
    elements.append(Paragraph("Disclaimer", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.darkblue, spaceAfter=10))
    disclaimer_text = "This is an automated report. For legal decisions, consult a certified forensic document examiner."
    elements.append(Paragraph(disclaimer_text, normal_style))

    doc.build(elements)
    print(f"✅ Report saved as {report_name}")
    return report_name


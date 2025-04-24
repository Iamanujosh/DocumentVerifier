from PIL import Image
import os
import tempfile
import os
from datetime import datetime
from django.http import HttpResponse, JsonResponse
import json
from django.shortcuts import render, redirect, get_object_or_404
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
import PyPDF2
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import kagglehub
from .models import Profile, DocumentReport, ChatMessage
from django.http import HttpResponse
from io import BytesIO
import tempfile
import os
import google.generativeai as genai
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from .models import StoredDocument
import json
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

genai.configure(api_key="AIzaSyAqzwDW-csXMz0Teihqns7Jtx5rBD4r-LE")  # Replace with your API key
model_gemini = genai.GenerativeModel('gemini-2.0-flash')

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
    docs = StoredDocument.objects.filter(user=request.user)
    return render(request, 'verifier/profile.html', {'documents': docs})
    


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
            report_path = generate_verification_report(request.user,temp_path, proba[0], result)
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


def generate_verification_report(user, image_path, prediction_proba, result):
    buffer = BytesIO()
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
     # Move buffer to the beginning
    buffer.seek(0)

    # Create a StoredDocument and store both the uploaded file and the generated report
    stored = StoredDocument(user=user)
    stored.report_file.save("document_verification_report.pdf", ContentFile(buffer.read()))
    stored.save()

    buffer.close()
    print(f"✅ Report saved as {report_name}")
    return report_name

@login_required
def report_list(request):
    reports = DocumentReport.objects.filter(user=request.user)
    return render(request, "verifier/report_list.html", {"reports": reports})

@login_required
def delete_report(request, report_id):
    report = get_object_or_404(DocumentReport, id=report_id, user=request.user)
    report.report_file.delete()
    report.delete()
    return redirect('report_list')

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        text = f"Error extracting text: {e}"
    return text

@login_required
def chatbot_view(request, report_id=None):
    import logging
    logger = logging.getLogger(__name__)
    
    report = None
    report_text = None
    messages = []
    
    try:
        # Get current report if specified
        if report_id:
            logger.info(f"Accessing report with ID: {report_id} for user: {request.user.username}")
            try:
                report = get_object_or_404(DocumentReport, id=report_id, user=request.user)
                report_path = report.report_file.path
                report_text = extract_text_from_pdf(report_path)
                # Get existing chat messages for this report
                messages = ChatMessage.objects.filter(report=report).order_by('created_at')
                # Store current report ID in session
                request.session['current_report_id'] = report_id
            except Exception as e:
                logger.error(f"Error accessing report {report_id}: {str(e)}")
                return render(request, "verifier/chatbot.html", {
                    "error": f"Could not access the report: {str(e)}"
                })
        
        # Helper function to determine prompt based on message content
        def create_appropriate_prompt(user_query, report_text):
            # Basic conversational phrases that don't need report analysis
            conversational_phrases = [
                "thank", "ok", "hello", "hi", "hey", "goodbye", "bye", 
                "thanks", "appreciate", "got it", "understood", "great", 
                "awesome", "cool", "nice"
            ]
            
            # Check if query is just conversational
            is_conversational = any(phrase in user_query.lower() for phrase in conversational_phrases) and len(user_query.split()) < 5
            
            if is_conversational:
                return f"User said: '{user_query}'. Respond conversationally without analyzing any report."
            elif report_text:
                return f"The user has a document with the following content: {report_text}\n\nUser question: {user_query}\n\nAnswer the user question clearly and concisely in a friendly, helpful tone. Focus on information from the report if relevant to the question."
            else:
                return f"User question: {user_query}. The user hasn't uploaded a report yet, so kindly let them know you need a report to analyze specific document content."
        
        # Handle AJAX request for chatbot
        if request.method == "POST" and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            try:
                data = json.loads(request.body)
                user_query = data.get("message", "")
                logger.info(f"AJAX chatbot request from user: {request.user.username}, query: {user_query[:50]}...")
                
                # Save user message
                user_message = ChatMessage(
                    report=report,
                    user=request.user,
                    content=user_query,
                    is_user=True
                )
                user_message.save()
                
                # Generate AI response
                try:
                    prompt = create_appropriate_prompt(user_query, report_text)
                    logger.debug(f"Sending prompt to model for user: {request.user.username}")
                    response = model_gemini.generate_content(prompt)
                    ai_response = response.text
                    
                except Exception as e:
                    logger.error(f"Model error for user {request.user.username}: {str(e)}")
                    ai_response = f"I'm sorry, I encountered an error while processing your request. Please try again."
                
                # Save AI response
                ai_message = ChatMessage(
                    report=report,
                    user=request.user,
                    content=ai_response,
                    is_user=False
                )
                ai_message.save()
                
                return JsonResponse({"response": ai_response})
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in AJAX request: {str(e)}")
                return JsonResponse({"error": "Invalid JSON"}, status=400)
            except Exception as e:
                logger.error(f"Unexpected error in AJAX handler: {str(e)}")
                return JsonResponse({"error": str(e)}, status=500)
        
        # Handle form submission for message
        if request.method == "POST" and request.POST.get("user_query"):
            user_query = request.POST["user_query"]
            logger.info(f"Form chatbot request from user: {request.user.username}, query: {user_query[:50]}...")
            
            # Save user message
            user_message = ChatMessage(
                report=report,
                user=request.user,
                content=user_query,
                is_user=True
            )
            user_message.save()
            
            # Generate AI response
            try:
                prompt = create_appropriate_prompt(user_query, report_text)
                logger.debug(f"Sending prompt to model for user: {request.user.username}")
                response = model_gemini.generate_content(prompt)
                ai_response = response.text
                
                # Save AI response
                ai_message = ChatMessage(
                    report=report,
                    user=request.user,
                    content=ai_response,
                    is_user=False
                )
                ai_message.save()
            except Exception as e:
                logger.error(f"Error generating AI response for user {request.user.username}: {str(e)}")
                # Save error message
                error_message = f"I'm sorry, I encountered an error while processing your request. Please try again."
                ai_message = ChatMessage(
                    report=report,
                    user=request.user,
                    content=error_message,
                    is_user=False
                )
                ai_message.save()
            
            # Refresh messages
            if report:
                messages = ChatMessage.objects.filter(report=report).order_by('created_at')
            
            # Return to the same page to continue the conversation
            return redirect(request.path)
        
        # Handle file upload
        if request.method == "POST" and request.FILES.get("document"):
            uploaded_file = request.FILES["document"]
            logger.info(f"File upload attempt by user: {request.user.username}, filename: {uploaded_file.name}")
            
            try:
                # Create new report
                new_report = DocumentReport(
                    user=request.user,
                    report_file=uploaded_file,
                    # title=uploaded_file.name
                )
                new_report.save()
                
                # Extract text
                report_text = extract_text_from_pdf(new_report.report_file.path)
                logger.info(f"Successfully processed file for user: {request.user.username}, report ID: {new_report.id}")
                
                # Redirect to the new report's chatbot view
                return redirect('chatbot_view', report_id=new_report.id)
            except Exception as e:
                logger.error(f"File upload/processing error for user {request.user.username}: {str(e)}")
                return render(request, "verifier/chatbot.html", {
                    "error": f"Error processing your file: {str(e)}"
                })
    
    except Exception as e:
        logger.error(f"Unhandled exception in chatbot_view: {str(e)}")
        return render(request, "verifier/chatbot.html", {
            "error": "An unexpected error occurred. Please try again or contact support."
        })
    
    return render(request, "verifier/chatbot.html", {
        "report": report, 
        "report_text": report_text,
        "messages": messages
    })

@csrf_exempt
def store_document_view(request):
    if request.method == 'POST' and request.user.is_authenticated:
        data = json.loads(request.body)
        if data.get('store'):
            # Assuming you already have a file uploaded and want to save a copy of it
            user = request.user

            # Replace this with the logic to get user's latest uploaded document
            # For demo, you can hardcode a file path or load from DB
            doc = StoredDocument.objects.create(
                user=user,  # or dynamically selected
                report='reports/sample_report.pdf'
            )
            return JsonResponse({"message": "Document and report stored."})
        else:
            return JsonResponse({"message": "Checkbox unchecked. Nothing stored."})
    return JsonResponse({"error": "Unauthorized or invalid request."}, status=400)
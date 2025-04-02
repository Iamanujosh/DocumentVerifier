from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home/Dashboard
    path('login/', views.login_view, name='login'),  # Login Page
    path('register/', views.register, name='register'),  # Register Page
    path('verify/', views.verify, name='verify'),  # File verify Page
    path('upload/', views.upload, name='upload'),  # file upload
    path('profile/', views.profile, name='profile'),  # 
    path('report/', views.report_file, name='report'), #result
    path('profile/upload-profile-picture/', views.upload_profile_picture, name='upload_profile_picture'),
    path('profile/delete-profile-picture/', views.delete_profile_picture, name='delete_profile_picture'),
    path('profile/upload-background/', views.upload_background, name='upload_background'),
    path('profile/delete-background/', views.delete_background, name='delete_background'),
]

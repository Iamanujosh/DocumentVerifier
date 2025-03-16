from django.urls import path
from . import views

urlpatterns = [
    path('', views.base_file, name='home'),  # Home/Dashboard
    path('login/', views.login_file, name='login'),  # Login Page
    
    path('register/', views.register_file, name='register'),  # Register Page
    path('upload/', views.upload_file, name='upload'),  # File Upload Page
    path('report/', views.report_file, name='report'),
    
    # path('logout/', views.user_logout, name='logout'),  # Logout
]

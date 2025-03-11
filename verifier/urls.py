from django.urls import path
from . import views

urlpatterns = [
    path('', views.base_file, name='dashboard'),  # Home/Dashboard
    # path('upload/', views.upload_file, name='upload'),  # File Upload Page
    # path('login/', views.user_login, name='login'),  # Login Page
    # path('register/', views.user_register, name='register'),  # Register Page
    # path('logout/', views.user_logout, name='logout'),  # Logout
]

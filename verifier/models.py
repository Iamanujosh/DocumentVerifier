from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_picture = models.ImageField(upload_to='verifier/profile_pictures/', blank=True, null=True)
    background_image = models.ImageField(upload_to='verifier/backgrounds/', blank=True, null=True)
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    default='verifier/profile_pictures/default.jpg'
    default='verifier/backgrounds/default.jpg'

    def __str__(self):
        return self.user.username

    
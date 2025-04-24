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
    
class DocumentReport(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    report_file = models.FileField(upload_to='reports/')
    created_at = models.DateTimeField(auto_now_add=True)

    def _str_(self):
        return f"Report for {self.user.username} - {self.created_at}"

class ChatMessage(models.Model):
    report = models.ForeignKey('DocumentReport', on_delete=models.CASCADE, null=True, related_name='messages')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    is_user = models.BooleanField(default=True)  # True for user messages, False for AI responses
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
    

class StoredDocument(models.Model):
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)  # or your custom user
     # assuming you're uploading a file
    report_file = models.FileField(upload_to='reports/', blank=True, null=True)
    stored_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Document by {self.user.username} on {self.stored_at.strftime('%Y-%m-%d')}"
    def _str_(self):
        return f"{'User' if self.is_user else 'AI'} message for {self.report}"
    
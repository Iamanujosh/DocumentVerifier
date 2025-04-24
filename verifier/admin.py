# admin.py
from django.contrib import admin
from .models import Profile,StoredDocument,DocumentReport,ChatMessage

admin.site.register(Profile)
admin.site.register(DocumentReport)
admin.site.register(ChatMessage)
admin.site.register(StoredDocument)

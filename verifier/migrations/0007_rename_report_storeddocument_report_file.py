# Generated by Django 5.1.5 on 2025-04-24 02:00

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('verifier', '0006_remove_storeddocument_document'),
    ]

    operations = [
        migrations.RenameField(
            model_name='storeddocument',
            old_name='report',
            new_name='report_file',
        ),
    ]

# Generated by Django 5.0.8 on 2024-10-02 09:30

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0002_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='recommendation',
            old_name='applied',
            new_name='is_applied',
        ),
    ]

# Generated by Django 5.0.8 on 2024-11-09 19:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('convochat', '0007_alter_sentimentcategory_options'),
    ]

    operations = [
        migrations.AddField(
            model_name='aitext',
            name='tool_calls',
            field=models.JSONField(blank=True, default=list),
        ),
    ]
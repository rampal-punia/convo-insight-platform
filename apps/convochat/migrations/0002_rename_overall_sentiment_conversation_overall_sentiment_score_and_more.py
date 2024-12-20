# Generated by Django 5.0.8 on 2024-10-02 08:57

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('convochat', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='conversation',
            old_name='overall_sentiment',
            new_name='overall_sentiment_score',
        ),
        migrations.AlterField(
            model_name='sentiment',
            name='granular_category',
            field=models.CharField(blank=True, choices=[('FR', 'Frustration'), ('SA', 'Satisfaction'), ('IN', 'Inquiry'), ('AN', 'Anger'), ('HA', 'Happiness'), ('CO', 'Confusion'), ('UR', 'Urgency')], max_length=2, null=True),
        ),
        migrations.AlterField(
            model_name='sentiment',
            name='message',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='sentiment', to='convochat.usertext'),
        ),
        migrations.AlterField(
            model_name='sentiment',
            name='score',
            field=models.FloatField(help_text='Sentiment score between -1 (very negative) and 1 (very positive)'),
        ),
    ]

# Generated by Django 5.0.8 on 2024-11-06 18:26

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('convochat', '0004_alter_topic_options_topic_category_topic_created_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='GranularEmotion',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, unique=True)),
                ('description', models.TextField(help_text='Description of this emotional category and when it applies')),
                ('is_active', models.BooleanField(default=True)),
                ('usage_count', models.PositiveIntegerField(default=0, help_text='Number of times this emotion was identified')),
            ],
        ),
        migrations.CreateModel(
            name='SentimentCategory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, unique=True)),
                ('description', models.TextField(help_text='Description of when this sentiment category should be used')),
                ('is_active', models.BooleanField(default=True)),
                ('priority_weight', models.FloatField(default=1.0, help_text='Weight factor for sentiment importance in analysis')),
            ],
        ),
        migrations.RemoveField(
            model_name='sentiment',
            name='granular_category',
        ),
        migrations.AddField(
            model_name='sentiment',
            name='confidence',
            field=models.FloatField(default=1.0, help_text="Model's confidence in this sentiment classification (0-1)"),
        ),
        migrations.AddField(
            model_name='sentiment',
            name='example_text',
            field=models.TextField(blank=True, help_text='Optional representative text to use as an example for this sentiment'),
        ),
        migrations.AddField(
            model_name='sentiment',
            name='is_example',
            field=models.BooleanField(default=False, help_text='Whether this sentiment can be used as an example for few-shot learning'),
        ),
        migrations.AlterField(
            model_name='sentiment',
            name='message',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='sentiment_analysis', to='convochat.usertext'),
        ),
        migrations.AlterField(
            model_name='sentiment',
            name='score',
            field=models.FloatField(help_text='Sentiment intensity score between -1 (very negative) and 1 (very positive)'),
        ),
        migrations.AddField(
            model_name='sentiment',
            name='granular_emotion',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='sentiments', to='convochat.granularemotion'),
        ),
        migrations.AddField(
            model_name='granularemotion',
            name='associated_sentiment',
            field=models.ForeignKey(help_text='The primary sentiment category this emotion is typically associated with', on_delete=django.db.models.deletion.CASCADE, related_name='emotions', to='convochat.sentimentcategory'),
        ),
        migrations.AlterField(
            model_name='sentiment',
            name='category',
            field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='sentiments', to='convochat.sentimentcategory'),
        ),
        migrations.AddIndex(
            model_name='sentiment',
            index=models.Index(fields=['category', 'score'], name='convochat_s_categor_aa4adb_idx'),
        ),
        migrations.AddIndex(
            model_name='sentiment',
            index=models.Index(fields=['is_example', 'confidence'], name='convochat_s_is_exam_73e29e_idx'),
        ),
    ]

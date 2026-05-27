"""Serializers for the NLP playground endpoints."""
from rest_framework import serializers


class NLPInputSerializer(serializers.Serializer):
    text = serializers.CharField(required=True)
    method = serializers.ChoiceField(
        choices=["finetuned", "few_shot_learning", "rag"],
        default="finetuned",
    )


class NLPResultSerializer(serializers.Serializer):
    label = serializers.JSONField()
    score = serializers.FloatField(required=False)
    explanation = serializers.CharField(required=False, allow_blank=True)
    category = serializers.CharField(required=False, allow_blank=True)


class NLPOutputSerializer(serializers.Serializer):
    task = serializers.CharField()
    method = serializers.CharField()
    result = NLPResultSerializer()

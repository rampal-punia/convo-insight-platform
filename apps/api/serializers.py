from rest_framework import serializers


class NLPInputSerializer(serializers.Serializer):
    text = serializers.CharField(required=True)
    method = serializers.ChoiceField(
        choices=['finetuned'],
        # choices=['finetuned', 'few_shot_learning', 'rag'],
        default='finetuned'
    )


class NLPOutputSerializer(serializers.Serializer):
    result = serializers.DictField()

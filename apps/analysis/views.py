# apps/analysis/views.py

from django.http import JsonResponse
from django.views.decorators.http import require_POST
from .models import Conversation
from .utils.performance_metric_calculation import process_conversation_metrics


@require_POST
def end_conversation(request, conversation_id):
    try:
        conversation = Conversation.objects.get(id=conversation_id)
        conversation.status = Conversation.Status.ENDED
        conversation.save()

        # Process metrics
        process_conversation_metrics(conversation_id)

        return JsonResponse({"status": "success", "message": "Conversation ended and metrics calculated"})
    except Conversation.DoesNotExist:
        return JsonResponse({"status": "error", "message": "Conversation not found"}, status=404)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

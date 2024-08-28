# apps/analysis/views.py

from .utils.agent_performance_evaluator import AgentPerformanceEvaluator
from convochat.models import Conversation
from analysis.models import LLMAgentPerformance
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import generic
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from .models import Conversation
from utils.performance_metric_calculation import process_conversation_metrics


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


class ConversationPerformanceView(LoginRequiredMixin, generic.DetailView):
    model = Conversation
    template_name = 'analysis/conversation_performance.html'
    context_object_name = 'conversation'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        conversation = self.get_object()

        evaluator = AgentPerformanceEvaluator()
        performance_data = evaluator.evaluate_conversation(conversation.id)

        context['performance_data'] = performance_data
        return context

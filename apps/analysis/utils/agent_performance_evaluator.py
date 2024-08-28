from typing import List, Dict
from django.db.models import Avg
from models import LLMAgentPerformance, Conversation


class AgentPerformanceEvaluator:
    def __init__(self):
        self.metrics = [
            'response_time',
            'accuracy_score',
            'relevance_score',
            'customer_satisfaction_score',
            'quality_score'
        ]
        self.weight_map = {
            'response_time': 0.2,
            'accuracy_score': 0.2,
            'relevance_score': 0.2,
            'customer_satisfaction_score': 0.2,
            'quality_score': 0.2
        }

    def evaluate_conversation(self, conversation_id: str) -> Dict:
        conversation = Conversation.objects.get(id=conversation_id)
        performances = LLMAgentPerformance.objects.filter(
            conversation=conversation)

        if not performances.exists():
            return {"error": "No performance data available for this conversation"}

        avg_metrics = performances.aggregate(
            **{f'avg_{metric}': Avg(metric) for metric in self.metrics}
        )

        overall_score = sum(
            avg_metrics[f'avg_{metric}'] * self.weight_map[metric]
            for metric in self.metrics
        )

        feedback = self._generate_feedback(avg_metrics)

        return {
            "conversation_id": conversation_id,
            "overall_score": overall_score,
            "metric_scores": avg_metrics,
            "feedback": feedback
        }

    def _generate_feedback(self, avg_metrics: Dict) -> List[str]:
        feedback = []

        response_time = avg_metrics['avg_response_time'].total_seconds()
        if response_time > 60:
            feedback.append(
                "Work on improving response time. Aim to respond within 60 seconds.")
        elif response_time < 30:
            feedback.append("Excellent response time! Keep up the good work.")

        if avg_metrics['avg_accuracy_score'] < 0.7:
            feedback.append(
                "Focus on improving the accuracy of your responses. Review and verify information before sending.")

        if avg_metrics['avg_relevance_score'] < 0.7:
            feedback.append(
                "Try to provide more relevant responses. Make sure you're addressing the user's specific questions or concerns.")

        if avg_metrics['avg_customer_satisfaction_score'] < 0.7:
            feedback.append(
                "Work on improving customer satisfaction. Show empathy and go the extra mile to resolve issues.")

        if avg_metrics['avg_quality_score'] < 0.7:
            feedback.append(
                "Focus on improving the overall quality of your responses. Ensure they are clear, concise, and helpful.")

        if not feedback:
            feedback.append(
                "Great job! Keep maintaining your high performance across all metrics.")

        return feedback


# Usage example
if __name__ == "__main__":
    evaluator = AgentPerformanceEvaluator()
    result = evaluator.evaluate_conversation("conversation_id_here")
    print(result)

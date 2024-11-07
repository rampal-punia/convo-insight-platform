# Modified Topic Model (convochat_models.py)
from django.db import transaction
from convochat.models import Topic
from django.core.management.base import BaseCommand
from typing import Any


class Command(BaseCommand):
    help = 'Create default topics for conversation analysis'

    def handle(self, *args: Any, **options: Any) -> str | None:
        default_topics = {
            # Product-Related Topics
            "Product Information": {
                "description": "Encompasses inquiries about product features, specifications, materials, dimensions, compatibility, and usage instructions.",
                "category": Topic.Category.PRODUCT,
                "priority_weight": 1.0
            },
            "Stock & Availability": {
                "description": "Handles inventory-related questions, stock alerts, pre-orders, and restock timelines.",
                "category": Topic.Category.PRODUCT,
                "priority_weight": 0.8
            },
            "Personalization & Recommendations": {
                "description": "Covers personalized product suggestions, size recommendations, and style advice.",
                "category": Topic.Category.PRODUCT,
                "priority_weight": 0.7
            },
            # Order Management Topics
            "Order Processing & Tracking": {
                "description": "Addresses order status inquiries, order modifications, and tracking information.",
                "category": Topic.Category.ORDER,
                "priority_weight": 1.0
            },
            "Shipping & Delivery": {
                "description": "Handles shipping methods, delivery timeframes, and shipping-related issues.",
                "category": Topic.Category.ORDER,
                "priority_weight": 0.9
            },
            "Returns & Refunds": {
                "description": "Manages product returns, refund requests, and return policy inquiries.",
                "category": Topic.Category.ORDER,
                "priority_weight": 0.9
            },
            # Payment & Account Topics
            "Payment & Billing": {
                "description": "Addresses payment methods, transactions, invoices, and billing issues.",
                "category": Topic.Category.PAYMENT,
                "priority_weight": 1.0
            },
            "Account Management": {
                "description": "Handles account creation, security, and profile maintenance issues.",
                "category": Topic.Category.PAYMENT,
                "priority_weight": 0.8
            },
            "Loyalty & Rewards": {
                "description": "Manages loyalty points, rewards, and membership benefits.",
                "category": Topic.Category.PAYMENT,
                "priority_weight": 0.7
            },
            # Customer Experience Topics
            "Technical Support": {
                "description": "Addresses website functionality, app issues, and technical problems.",
                "category": Topic.Category.EXPERIENCE,
                "priority_weight": 0.9
            },
            "Promotions & Discounts": {
                "description": "Handles promotional offers, discount codes, and sale inquiries.",
                "category": Topic.Category.EXPERIENCE,
                "priority_weight": 0.8
            },
            "Customer Feedback": {
                "description": "Manages customer reviews, suggestions, and general feedback.",
                "category": Topic.Category.EXPERIENCE,
                "priority_weight": 0.7
            },
        }

        with transaction.atomic():
            for name, details in default_topics.items():
                Topic.objects.update_or_create(
                    name=name,
                    defaults={
                        "description": details["description"],
                        "category": details["category"],
                        "priority_weight": details["priority_weight"],
                        "is_active": True
                    }
                )
                self.stdout.write(self.style.SUCCESS(
                    f'Topic created/updated: {name}'))

        self.stdout.write(self.style.SUCCESS(
            'Successfully generated topic data'))

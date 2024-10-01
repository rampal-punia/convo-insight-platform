# python manage.py create_topics

from typing import Any
from django.core.management.base import BaseCommand
from convochat.models import Topic


class Command(BaseCommand):
    help = 'Create topics'

    def handle(self, *args: Any, **options: Any) -> str | None:
        topics = {
            "Product Information": "Covers questions about product features, specifications, and usage.",
            "Order Status": "Relates to inquiries about current or past orders.",
            "Shipping and Delivery": "Addresses questions about shipping methods, times, and tracking.",
            "Returns and Refunds": "Covers policies and processes for returning items and getting refunds.",
            "Payment and Billing": "Includes issues related to payment methods, charges, and invoices.",
            "Technical Support": "For troubleshooting product issues or website/app functionality problems.",
            "Account Management": "Covers account creation, login issues, and profile updates.",
            "Product Availability": "Addresses stock inquiries, back-order information, and restocking timelines.",
            "Promotions and Discounts": "Covers questions about current sales, coupon codes, and loyalty programs.",
            "Customer Feedback": "For general comments, suggestions, and complaints about products or services.",
            "Size and Fit": "Specifically for apparel and footwear e-commerce, addressing sizing questions.",
            "Product Comparisons": "Helps customers choose between similar products.",
        }
        for name, description in topics.items():
            Topic.objects.create(
                name=name,
                description=description

            )
            self.stdout.write(self.style.SUCCESS(f'Topic created: {name}'))
        self.stdout.write(self.style.SUCCESS(
            'Successfully generated topic data'))

# python manage.py create_intents

from typing import Any
from django.core.management.base import BaseCommand, CommandParser
from convochat.models import Intent


class Command(BaseCommand):
    help = 'Create intents'

    def handle(self, *args: Any, **options: Any) -> str | None:
        intents = [
            "Ask Product Question",
            "File Complaint",
            "Seek Order Assistance",
            "Make Purchase",
            "Cancel Order/Service",
            "Provide Feedback",
            "Track Order",
            "Request Refund",
            "Inquire About Pricing",
            "General Inquiry",
            "Request Technical Support",
        ]
        for name in intents:
            Intent.objects.create(
                name=name,
                description=f"Description of the intent '{name}'"

            )
            self.stdout.write(self.style.SUCCESS(f'Intent created: {name}'))
        self.stdout.write(self.style.SUCCESS(
            'Successfully generated intent data'))

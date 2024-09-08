# python manage.py create_random_users 50

from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
# from django.utils.crypto import get_random_string

User = get_user_model()


class Command(BaseCommand):
    help = 'Create random users'

    def add_arguments(self, parser):
        parser.add_argument('total', type=int,
                            help='Indicates the number of users to be created')

    def handle(self, *args, **kwargs):
        total = kwargs['total']
        for _ in range(total):
            username = f'user_{_}'
            email = f'{username}@example.com'
            password = 'Aa#123456'
            User.objects.create_user(
                username=username, email=email, password=password)
            self.stdout.write(self.style.SUCCESS(f'User created: {username}'))

        self.stdout.write(self.style.SUCCESS(
            f'Successfully created {total} random users'))

# run 'python manage.py generate_dummy_data'

import random
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from products.models import Category, Product
from orders.models import Order, OrderItem

User = get_user_model()


class Command(BaseCommand):
    help = 'Generate dummy data for products and orders'

    def handle(self, *args, **options):
        # Create categories
        categories = ['Electronics', 'Clothing',
                      'Books', 'Home & Garden', 'Toys']
        for cat in categories:
            Category.objects.create(name=cat)

        # Create Products
        for _ in range(50):
            Product.objects.create(
                name=f"Product {_}",
                description=f"Description in detail for the Product {_}",
                price=random.uniform(10, 500),
                category=Category.objects.order_by('?').first(),
                stock=random.randint(0, 100)
            )

       # Create orders
        users = User.objects.all()
        if not users.count() > 2:
            self.stdout.write(self.style.WARNING(
                'No users found. Please run create_random_users command first.'))
            return

        for _ in range(100):
            user = random.choice(users)
            order = Order.objects.create(
                user=user,
                status=random.choice(['PE', 'PR', 'SH', 'DE', 'CA']),
                total_amount=0
            )

            # Add order items
            for _ in range(random.randint(1, 5)):
                product = Product.objects.order_by('?').first()
                quantity = random.randint(1, 3)
                OrderItem.objects.create(
                    order=order,
                    product=product,
                    quantity=quantity,
                    price=product.price
                )

            # Update total amount
            order.total_amount = sum(
                item.price * item.quantity for item in order.items.all())
            order.save()

        self.stdout.write(self.style.SUCCESS(
            'Successfully generated dummy data'))

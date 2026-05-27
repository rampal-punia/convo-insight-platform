"""Seed demo data for local dev / intern onboarding.

Usage::

    python manage.py seed_demo               # default sizes
    python manage.py seed_demo --reset       # wipe demo data first
    python manage.py seed_demo --users 20    # custom counts

Creates: categories, products, demo users, orders + items + tracking events.
"""

from __future__ import annotations

import random
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from orders.models import Order, OrderItem, OrderTracking
from products.models import Category, Product

User = get_user_model()

DEMO_PASSWORD = "demo12345"  # noqa: S105 - dev-only fixture
DEMO_TAG = "[demo]"

CATEGORIES = [
    ("Laptops", "Portable computers for work and play."),
    ("Smartphones", "Modern handsets and accessories."),
    ("Audio", "Headphones, speakers, and microphones."),
    ("Wearables", "Smartwatches and fitness bands."),
    ("Accessories", "Cables, chargers, and add-ons."),
]

PRODUCTS = [
    ("Aparsoft Air 14", "Laptops", 1299, 25),
    ("Aparsoft Pro 16", "Laptops", 2199, 10),
    ("Aparsoft Phone X", "Smartphones", 899, 40),
    ("Aparsoft Phone Lite", "Smartphones", 499, 60),
    ("BoomBuds Pro", "Audio", 199, 80),
    ("StudioMic One", "Audio", 149, 35),
    ("PulseBand 3", "Wearables", 129, 50),
    ("FocusWatch S2", "Wearables", 249, 30),
    ("USB-C Hub", "Accessories", 49, 200),
    ("Fast Charger 65W", "Accessories", 39, 150),
]


class Command(BaseCommand):
    help = "Populate the database with demo data for development & intern onboarding."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset", action="store_true", help="Delete existing demo data first."
        )
        parser.add_argument(
            "--users", type=int, default=8, help="Number of demo users to create."
        )
        parser.add_argument(
            "--orders", type=int, default=25, help="Number of demo orders to create."
        )

    @transaction.atomic
    def handle(self, *args, **opts):
        if opts["reset"]:
            self._reset()

        cat_map = self._seed_categories()
        products = self._seed_products(cat_map)
        users = self._seed_users(opts["users"])
        self._seed_orders(users, products, opts["orders"])

        self.stdout.write(self.style.SUCCESS("\nDemo data seeded successfully."))
        self.stdout.write(f"  Categories: {Category.objects.count()}")
        self.stdout.write(f"  Products:   {Product.objects.count()}")
        self.stdout.write(f"  Users:      {User.objects.count()}")
        self.stdout.write(f"  Orders:     {Order.objects.count()}")
        self.stdout.write(
            f"\nLog in as any demo user with password: {self.style.WARNING(DEMO_PASSWORD)}\n"
        )

    # ----------------------------- helpers ----------------------------- #

    def _reset(self):
        self.stdout.write("Deleting existing demo data...")
        OrderTracking.objects.filter(order__user__username__startswith="demo_").delete()
        OrderItem.objects.filter(order__user__username__startswith="demo_").delete()
        Order.objects.filter(user__username__startswith="demo_").delete()
        User.objects.filter(username__startswith="demo_").delete()

    def _seed_categories(self) -> dict[str, Category]:
        result = {}
        for name, desc in CATEGORIES:
            obj, _ = Category.objects.get_or_create(
                name=name, defaults={"description": desc}
            )
            result[name] = obj
        return result

    def _seed_products(self, cat_map: dict[str, Category]) -> list[Product]:
        products: list[Product] = []
        for name, cat_name, price, stock in PRODUCTS:
            obj, _ = Product.objects.get_or_create(
                name=name,
                defaults={
                    "description": f"{name} - {DEMO_TAG}",
                    "price": Decimal(price),
                    "category": cat_map[cat_name],
                    "stock": stock,
                },
            )
            products.append(obj)
        return products

    def _seed_users(self, n: int) -> list:
        users = []
        for i in range(1, n + 1):
            username = f"demo_user_{i:02d}"
            user, created = User.objects.get_or_create(
                username=username,
                defaults={
                    "email": f"{username}@example.com",
                    "first_name": f"Demo{i}",
                    "last_name": "User",
                },
            )
            if created:
                user.set_password(DEMO_PASSWORD)
                user.save()
            users.append(user)
        admin, created = User.objects.get_or_create(
            username="demo_admin",
            defaults={
                "email": "demo_admin@example.com",
                "is_staff": True,
                "is_superuser": True,
            },
        )
        if created:
            admin.set_password(DEMO_PASSWORD)
            admin.save()
        users.append(admin)
        return users

    def _seed_orders(self, users: list, products: list[Product], n: int):
        statuses = [s[0] for s in Order.Status.choices]
        for _ in range(n):
            user = random.choice(users)
            order = Order.objects.create(
                user=user,
                status=random.choice(statuses),
                shipping_method=random.choice(
                    [s[0] for s in Order.ShippingMethod.choices]
                ),
                tracking_number=f"TRK{random.randint(100000, 999999)}",
                delivery_address="221B Baker Street, London",
                carrier=random.choice(["FedEx", "UPS", "DHL", "BlueDart"]),
                estimated_delivery=timezone.now().date(),
            )
            total = Decimal("0")
            for product in random.sample(products, k=random.randint(1, 3)):
                qty = random.randint(1, 3)
                OrderItem.objects.create(
                    order=order, product=product, quantity=qty, price=product.price
                )
                total += product.price * qty
            order.total_amount = total
            order.save(update_fields=["total_amount"])

            for _ in range(random.randint(1, 3)):
                order.add_tracking_update(
                    status=random.choice(["PE", "PR", "SH", "TR", "DE"]),
                    location=random.choice(
                        ["Warehouse", "Hub", "City Office", "Out for delivery"]
                    ),
                    description="Auto-generated demo event",
                )

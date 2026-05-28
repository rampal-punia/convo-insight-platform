"""Unit tests for the Orders app models."""

import pytest
from decimal import Decimal
from django.contrib.auth import get_user_model

from orders.models import Order, OrderItem, OrderTracking, OrderConversationLink
from products.models import Category, Product

pytestmark = pytest.mark.django_db

User = get_user_model()


@pytest.fixture
def user():
    return User.objects.create_user(username='buyer', password='pw12345')


@pytest.fixture
def category():
    return Category.objects.create(name='Electronics')


@pytest.fixture
def product(category):
    return Product.objects.create(
        name='Laptop', description='Test', price=Decimal('999.99'), category=category, stock=10,
    )


class TestOrderModel:
    def test_create_order(self, user):
        order = Order.objects.create(user=user, total_amount=Decimal('100.00'))
        assert order.pk is not None
        assert order.status == Order.Status.PENDING
        assert 'Order' in str(order)
        assert 'buyer' in str(order)

    def test_order_status_choices(self):
        expected = {'PE', 'PR', 'SH', 'TR', 'DE', 'RT', 'RF', 'CA'}
        actual = {c[0] for c in Order.Status.choices}
        assert actual == expected

    def test_order_shipping_method_choices(self):
        expected = {'ST', 'EX', 'ON', 'LO'}
        actual = {c[0] for c in Order.ShippingMethod.choices}
        assert actual == expected

    def test_order_default_shipping(self, user):
        order = Order.objects.create(user=user)
        assert order.shipping_method == Order.ShippingMethod.STANDARD

    def test_order_auto_timestamps(self, user):
        order = Order.objects.create(user=user)
        assert order.created is not None
        assert order.modified is not None

    def test_get_tracking_status_no_history(self, user):
        order = Order.objects.create(user=user)
        assert order.get_tracking_status() == Order.Status.PENDING

    def test_get_tracking_status_with_history(self, user):
        order = Order.objects.create(user=user)
        OrderTracking.objects.create(order=order, status='SH')
        OrderTracking.objects.create(order=order, status='TR')
        assert order.get_tracking_status() == 'TR'  # latest

    def test_get_shipping_timeline(self, user):
        order = Order.objects.create(user=user)
        OrderTracking.objects.create(order=order, status='PL')
        OrderTracking.objects.create(order=order, status='SH')
        timeline = order.get_shipping_timeline()
        assert timeline.count() == 2
        assert list(timeline.values_list('status', flat=True)) == ['PL', 'SH']

    def test_add_tracking_update(self, user):
        order = Order.objects.create(user=user)
        tracking = order.add_tracking_update('SH', location='Warehouse', description='Shipped')
        assert tracking.pk is not None
        assert tracking.status == 'SH'
        assert tracking.location == 'Warehouse'


class TestOrderItemModel:
    def test_create_order_item(self, user, product):
        order = Order.objects.create(user=user)
        item = OrderItem.objects.create(order=order, product=product, quantity=2, price=product.price)
        assert item.pk is not None
        assert str(item) == f'2 x {product.name} in Order {order.id}'


class TestOrderTrackingModel:
    def test_create_tracking(self, user):
        order = Order.objects.create(user=user)
        t = OrderTracking.objects.create(order=order, status='PL', description='Order placed')
        assert t.pk is not None
        assert 'Tracking update' in str(t)

    def test_tracking_ordering(self, user):
        order = Order.objects.create(user=user)
        OrderTracking.objects.create(order=order, status='PL')
        OrderTracking.objects.create(order=order, status='SH')
        trackings = list(OrderTracking.objects.values_list('status', flat=True))
        assert trackings == ['SH', 'PL']  # -timestamp ordering

    def test_tracking_indexes(self):
        indexes = [idx.fields for idx in OrderTracking._meta.indexes]
        assert any('order' in str(idx) and 'timestamp' in str(idx) for idx in indexes)

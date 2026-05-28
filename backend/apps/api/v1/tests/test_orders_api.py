"""API tests for Orders endpoints."""

import pytest
from decimal import Decimal

from orders.models import Order, OrderItem, OrderTracking
from products.models import Category, Product

pytestmark = pytest.mark.django_db


@pytest.fixture
def category():
    return Category.objects.create(name='Electronics')


@pytest.fixture
def product(category):
    return Product.objects.create(
        name='Laptop', description='Test', price=Decimal('999.99'), category=category, stock=10,
    )


@pytest.fixture
def order(user, product):
    order = Order.objects.create(user=user, total_amount=Decimal('999.99'), status=Order.Status.PENDING)
    OrderItem.objects.create(order=order, product=product, quantity=1, price=product.price)
    return order


@pytest.fixture
def staff_order(staff_user, product):
    order = Order.objects.create(
        user=staff_user, total_amount=Decimal('1999.98'), status=Order.Status.PROCESSING,
    )
    OrderItem.objects.create(order=order, product=product, quantity=2, price=product.price)
    return order


class TestOrderCRUD:
    def test_list_own_orders(self, auth_client, order):
        resp = auth_client.get('/api/v1/orders/')
        assert resp.status_code == 200
        assert resp.data['count'] == 1

    def test_create_order(self, auth_client, user, product):
        resp = auth_client.post('/api/v1/orders/', {
            'total_amount': '99.99',
            'shipping_method': 'EX',
            'delivery_address': '123 Main St',
        })
        assert resp.status_code == 201
        assert resp.data['status'] == 'PE'
        assert Order.objects.filter(user=user).count() == 1

    def test_retrieve_order_detail(self, auth_client, order):
        resp = auth_client.get(f'/api/v1/orders/{order.pk}/')
        assert resp.status_code == 200
        assert resp.data['status'] == 'PE'
        assert len(resp.data['items']) == 1
        assert resp.data['items'][0]['product_name'] == 'Laptop'

    def test_non_staff_sees_own_orders_only(self, auth_client, order, staff_order):
        resp = auth_client.get('/api/v1/orders/')
        assert resp.status_code == 200
        assert resp.data['count'] == 1
        assert resp.data['results'][0]['id'] == order.pk

    def test_staff_sees_all_orders(self, staff_client, order, staff_order):
        resp = staff_client.get('/api/v1/orders/')
        assert resp.status_code == 200
        assert resp.data['count'] == 2


class TestOrderCancel:
    def test_cancel_pending_order(self, auth_client, order):
        resp = auth_client.post(f'/api/v1/orders/{order.pk}/cancel/')
        assert resp.status_code == 200
        assert resp.data['status'] == 'CA'
        assert OrderTracking.objects.filter(order=order).count() == 1

    def test_cancel_processing_order(self, auth_client, order):
        order.status = Order.Status.PROCESSING
        order.save()
        resp = auth_client.post(f'/api/v1/orders/{order.pk}/cancel/')
        assert resp.status_code == 200
        assert resp.data['status'] == 'CA'

    def test_cannot_cancel_delivered(self, auth_client, order):
        order.status = Order.Status.DELIVERED
        order.save()
        resp = auth_client.post(f'/api/v1/orders/{order.pk}/cancel/')
        assert resp.status_code == 400

    def test_cannot_cancel_already_cancelled(self, auth_client, order):
        order.status = Order.Status.CANCELLED
        order.save()
        resp = auth_client.post(f'/api/v1/orders/{order.pk}/cancel/')
        assert resp.status_code == 400


class TestOrderRefund:
    def test_refund_delivered_order(self, auth_client, order):
        order.status = Order.Status.DELIVERED
        order.save()
        resp = auth_client.post(f'/api/v1/orders/{order.pk}/refund/')
        assert resp.status_code == 200
        assert resp.data['status'] == 'RF'
        assert OrderTracking.objects.filter(order=order).exists()

    def test_refund_with_reason(self, auth_client, order):
        order.status = Order.Status.DELIVERED
        order.save()
        resp = auth_client.post(f'/api/v1/orders/{order.pk}/refund/', {'reason': 'Defective item'})
        assert resp.status_code == 200
        assert resp.data['status'] == 'RF'

    def test_refund_shipped_order(self, auth_client, order):
        order.status = Order.Status.SHIPPED
        order.save()
        resp = auth_client.post(f'/api/v1/orders/{order.pk}/refund/')
        assert resp.status_code == 200
        assert resp.data['status'] == 'RF'

    def test_cannot_refund_pending(self, auth_client, order):
        resp = auth_client.post(f'/api/v1/orders/{order.pk}/refund/')
        assert resp.status_code == 400

    def test_cannot_refund_cancelled(self, auth_client, order):
        order.status = Order.Status.CANCELLED
        order.save()
        resp = auth_client.post(f'/api/v1/orders/{order.pk}/refund/')
        assert resp.status_code == 400


class TestOrderTracking:
    def test_get_tracking(self, auth_client, order):
        OrderTracking.objects.create(order=order, status='PL', description='Placed')
        resp = auth_client.get(f'/api/v1/orders/{order.pk}/tracking/')
        assert resp.status_code == 200
        assert len(resp.data) == 1

    def test_add_tracking(self, auth_client, order):
        resp = auth_client.post(f'/api/v1/orders/{order.pk}/tracking/add/', {
            'status': 'SH',
            'description': 'Shipped from warehouse',
        })
        assert resp.status_code == 201
        assert resp.data['status'] == 'SH'

    def test_order_items(self, auth_client, order):
        resp = auth_client.get(f'/api/v1/orders/{order.pk}/items/')
        assert resp.status_code == 200
        assert len(resp.data) == 1
        assert resp.data[0]['product_name'] == 'Laptop'


class TestStaffActions:
    def test_mark_shipped(self, staff_client, staff_order):
        resp = staff_client.post(f'/api/v1/orders/{staff_order.pk}/mark-shipped/')
        assert resp.status_code == 200
        assert resp.data['status'] == 'SH'

    def test_non_staff_cannot_mark_shipped(self, auth_client, staff_order):
        resp = auth_client.post(f'/api/v1/orders/{staff_order.pk}/mark-shipped/')
        assert resp.status_code == 403

    def test_update_status(self, staff_client, staff_order):
        resp = staff_client.post(f'/api/v1/orders/{staff_order.pk}/update-status/', {'status': 'PR'})
        assert resp.status_code == 200
        assert resp.data['status'] == 'PR'

    def test_update_status_invalid(self, staff_client, staff_order):
        resp = staff_client.post(f'/api/v1/orders/{staff_order.pk}/update-status/', {'status': 'ZZ'})
        assert resp.status_code == 400

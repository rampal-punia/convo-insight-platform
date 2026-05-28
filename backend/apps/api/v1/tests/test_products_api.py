"""API tests for Products endpoints."""

import pytest
from decimal import Decimal

from products.models import Category, Product

pytestmark = pytest.mark.django_db


@pytest.fixture
def category():
    return Category.objects.create(name='Electronics', description='Gadgets')


@pytest.fixture
def sample_products(category):
    return [
        Product.objects.create(name='Laptop', description='Pro laptop', price=Decimal('1200'), category=category, stock=100),
        Product.objects.create(name='Mouse', description='Wireless mouse', price=Decimal('25'), category=category, stock=3),
        Product.objects.create(name='Keyboard', description='Mechanical keyboard', price=Decimal('80'), category=category, stock=0),
        Product.objects.create(name='Monitor', description='4K monitor', price=Decimal('400'), category=category, stock=60),
    ]


class TestProductCRUD:
    def test_list_products(self, auth_client, sample_products):
        resp = auth_client.get('/api/v1/products/')
        assert resp.status_code == 200
        assert resp.data['count'] == 4

    def test_create_product(self, auth_client, category):
        resp = auth_client.post('/api/v1/products/', {
            'name': 'Webcam',
            'description': 'HD webcam',
            'price': '49.99',
            'category': category.pk,
            'stock': 15,
        })
        assert resp.status_code == 201
        assert resp.data['name'] == 'Webcam'
        assert resp.data['stock'] == 15

    def test_retrieve_product(self, auth_client, sample_products):
        resp = auth_client.get(f'/api/v1/products/{sample_products[0].pk}/')
        assert resp.status_code == 200
        assert resp.data['name'] == 'Laptop'
        assert resp.data['category_name'] == 'Electronics'

    def test_update_product(self, auth_client, sample_products):
        resp = auth_client.patch(f'/api/v1/products/{sample_products[0].pk}/', {'stock': 200})
        assert resp.status_code == 200
        assert resp.data['stock'] == 200

    def test_delete_product(self, auth_client, sample_products):
        resp = auth_client.delete(f'/api/v1/products/{sample_products[0].pk}/')
        assert resp.status_code == 204
        assert Product.objects.count() == 3

    def test_search_products(self, auth_client, sample_products):
        resp = auth_client.get('/api/v1/products/?search=laptop')
        assert resp.status_code == 200
        assert resp.data['count'] == 1
        assert resp.data['results'][0]['name'] == 'Laptop'

    def test_filter_by_category(self, auth_client, sample_products, category):
        resp = auth_client.get(f'/api/v1/products/?category={category.pk}')
        assert resp.status_code == 200
        assert resp.data['count'] == 4


class TestProductActions:
    def test_in_stock(self, auth_client, sample_products):
        resp = auth_client.get('/api/v1/products/in-stock/')
        assert resp.status_code == 200
        names = {p['name'] for p in resp.data['results']}
        assert 'Keyboard' not in names  # stock=0
        assert 'Laptop' in names

    def test_low_stock_default_threshold(self, auth_client, sample_products):
        resp = auth_client.get('/api/v1/products/low-stock/')
        assert resp.status_code == 200
        names = {p['name'] for p in resp.data['results']}
        assert 'Mouse' in names  # stock=3 <= 5
        assert 'Keyboard' in names  # stock=0 <= 5

    def test_low_stock_custom_threshold(self, auth_client, sample_products):
        resp = auth_client.get('/api/v1/products/low-stock/?threshold=50')
        assert resp.status_code == 200
        names = {p['name'] for p in resp.data['results']}
        assert 'Mouse' in names  # stock=3
        assert 'Keyboard' in names  # stock=0

    def test_featured_products(self, auth_client, sample_products):
        resp = auth_client.get('/api/v1/products/featured/')
        assert resp.status_code == 200
        names = {p['name'] for p in resp.data['results']}
        assert 'Laptop' in names  # stock=100 > 50
        assert 'Monitor' in names  # stock=60 > 50
        assert 'Mouse' not in names  # stock=3
        assert 'Keyboard' not in names  # stock=0


class TestCategoryCRUD:
    def test_list_categories(self, auth_client, category):
        resp = auth_client.get('/api/v1/categories/')
        assert resp.status_code == 200
        assert resp.data['count'] == 1
        assert resp.data['results'][0]['product_count'] == 0

    def test_create_category(self, auth_client):
        resp = auth_client.post('/api/v1/categories/', {'name': 'Books', 'description': 'All books'})
        assert resp.status_code == 201
        assert resp.data['name'] == 'Books'

    def test_category_products_action(self, auth_client, category, sample_products):
        resp = auth_client.get(f'/api/v1/categories/{category.pk}/products/')
        assert resp.status_code == 200
        assert len(resp.data['results']) == 4

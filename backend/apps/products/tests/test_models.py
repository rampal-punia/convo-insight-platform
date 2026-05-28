"""Unit tests for the Products app models."""

import pytest
from decimal import Decimal

from products.models import Category, Product

pytestmark = pytest.mark.django_db


class TestCategoryModel:
    def test_create_category(self):
        cat = Category.objects.create(name='Electronics', description='Gadgets and devices')
        assert cat.pk is not None
        assert str(cat) == 'Electronics'

    def test_category_verbose_name_plural(self):
        assert Category._meta.verbose_name_plural == 'Categories'

    def test_category_description_blank(self):
        cat = Category.objects.create(name='Books')
        assert cat.description == ''


class TestProductModel:
    @pytest.fixture
    def category(self):
        return Category.objects.create(name='Electronics')

    def test_create_product(self, category):
        p = Product.objects.create(
            name='Laptop',
            description='A powerful laptop',
            price=Decimal('999.99'),
            category=category,
            stock=10,
        )
        assert p.pk is not None
        assert str(p) == 'Laptop'
        assert p.stock == 10

    def test_product_default_stock(self, category):
        p = Product.objects.create(
            name='Mouse',
            description='Wireless mouse',
            price=Decimal('29.99'),
            category=category,
        )
        assert p.stock == 0

    def test_product_auto_timestamps(self, category):
        p = Product.objects.create(
            name='Keyboard',
            description='Mechanical keyboard',
            price=Decimal('79.99'),
            category=category,
        )
        assert p.created is not None
        assert p.modified is not None

    def test_product_ordering(self, category):
        Product.objects.create(name='B', price=Decimal('10'), category=category, stock=1)
        Product.objects.create(name='A', price=Decimal('20'), category=category, stock=2)
        products = list(Product.objects.values_list('name', flat=True))
        assert len(products) == 2
        assert set(products) == {'A', 'B'}

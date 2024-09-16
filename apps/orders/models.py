# orders/models.py
from django.db import models
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _
from products.models import Product

from config.models import CreationModificationDateBase

User = get_user_model()


class Order(CreationModificationDateBase):
    class Status(models.TextChoices):
        PROCESSESING = 'PR', _('Processing')
        SHIPPED = 'SH', _('Shipped')
        PENDING = 'PE', _('Pending')  # Make it 'transit'
        DELIVERED = 'DE', _('Delivered')
        CANCELLED = 'CA', _('Cancelled')

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    status = models.CharField(
        max_length=2,
        choices=Status.choices,
        default=Status.PENDING)  # default 'processing'
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return f"Order {self.id} by {self.user.username}"


class OrderItem(models.Model):
    order = models.ForeignKey(
        Order, related_name='items', on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.IntegerField(default=1)
    price = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return f"{self.quantity} x {self.product.name} in Order {self.order.id}"

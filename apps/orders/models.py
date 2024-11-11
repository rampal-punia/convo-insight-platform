# orders/models.py
from django.db import models
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _
from products.models import Product

from config.models import CreationModificationDateBase

User = get_user_model()


class Order(CreationModificationDateBase):
    class Status(models.TextChoices):
        PENDING = 'PE', _('Pending')
        PROCESSING = 'PR', _('Processing')
        SHIPPED = 'SH', _('Shipped')
        IN_TRANSIT = 'TR', _('In Transit')
        DELIVERED = 'DE', _('Delivered')
        RETURNED = 'RT', _('Returned')
        REFUNDED = 'RF', _('Refunded')
        CANCELLED = 'CA', _('Cancelled')

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    status = models.CharField(
        max_length=2,
        choices=Status.choices,
        default=Status.PENDING)
    total_amount = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True
    )

    class Meta:
        ordering = ['-created']

    def __str__(self):
        return f"Order {self.id} by {self.user.username}"


class OrderItem(models.Model):
    order = models.ForeignKey(
        Order, related_name='items', on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.IntegerField(default=1)
    price = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True
    )

    def __str__(self):
        return f"{self.quantity} x {self.product.name} in Order {self.order.id}"


class OrderConversationLink(models.Model):
    order = models.ForeignKey(
        'Order', on_delete=models.CASCADE, related_name='conversation_links')
    conversation = models.ForeignKey(
        'convochat.Conversation',
        on_delete=models.CASCADE,
        related_name='order_links',
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('order', 'conversation')
        ordering = ['-created_at']

    def __str__(self):
        return f"Order {self.order.id} - Conversation {self.conversation.id}"

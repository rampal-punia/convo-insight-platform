# orders/models.py
from django.db import models
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _
from products.models import Product

from config.models import CreationModificationDateBase
from django.core.validators import RegexValidator

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

    class ShippingMethod(models.TextChoices):
        STANDARD = 'ST', _('Standard Shipping')
        EXPRESS = 'EX', _('Express Shipping')
        OVERNIGHT = 'ON', _('Overnight Shipping')
        LOCAL = 'LO', _('Local Delivery')

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
    tracking_number = models.CharField(
        max_length=50,
        null=True,
        blank=True,
        validators=[
            RegexValidator(
                regex=r'^[A-Za-z0-9-]+$',
                message='Tracking number can only contain letters, numbers, and hyphens'
            )
        ]
    )
    shipping_method = models.CharField(
        max_length=2,
        choices=ShippingMethod.choices,
        default=ShippingMethod.STANDARD
    )
    estimated_delivery = models.DateField(
        null=True,
        blank=True,
        help_text="Estimated delivery date"
    )
    shipped_date = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Date when the order was shipped"
    )
    delivery_address = models.TextField(
        null=True,
        blank=True,
        help_text="Shipping address for delivery"
    )
    carrier = models.CharField(
        max_length=50,
        null=True,
        blank=True,
        help_text="Shipping carrier (e.g., FedEx, UPS)"
    )

    class Meta:
        ordering = ['-created']

    def __str__(self):
        return f"Order {self.id} by {self.user.username}"

    def get_tracking_status(self):
        """Get the latest tracking status"""
        latest_tracking = self.tracking_history.order_by('-timestamp').first()
        return latest_tracking.status if latest_tracking else self.status

    def get_shipping_timeline(self):
        """Get ordered list of tracking events"""
        return self.tracking_history.order_by('timestamp').all()

    def add_tracking_update(self, status, location=None, description=None):
        """Add a new tracking update"""
        return OrderTracking.objects.create(
            order=self,
            status=status,
            location=location,
            description=description
        )


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


class OrderTracking(CreationModificationDateBase):
    """Model to store tracking history for orders"""
    class TrackingStatus(models.TextChoices):
        ORDER_PLACED = 'PL', _('Order Placed')
        PROCESSING = 'PR', _('Processing')
        PICKED = 'PK', _('Picked')
        PACKED = 'PA', _('Packed')
        SHIPPED = 'SH', _('Shipped')
        IN_TRANSIT = 'TR', _('In Transit')
        OUT_FOR_DELIVERY = 'OD', _('Out for Delivery')
        DELIVERED = 'DE', _('Delivered')
        FAILED_ATTEMPT = 'FA', _('Failed Delivery Attempt')
        EXCEPTION = 'EX', _('Delivery Exception')
        RETURNED = 'RT', _('Returned to Sender')

    order = models.ForeignKey(
        Order,
        related_name='tracking_history',
        on_delete=models.CASCADE
    )
    timestamp = models.DateTimeField(
        auto_now_add=True,
        help_text="When this tracking update occurred"
    )
    status = models.CharField(
        max_length=2,
        choices=TrackingStatus.choices,
        help_text="Current tracking status"
    )
    location = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="Location of the package"
    )
    description = models.TextField(
        null=True,
        blank=True,
        help_text="Additional details about this tracking update"
    )

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['order', '-timestamp']),
            models.Index(fields=['status', 'timestamp']),
        ]

    def __str__(self):
        return f"Tracking update for Order {self.order.id}: {self.get_status_display()}"


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

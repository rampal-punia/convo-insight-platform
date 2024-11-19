from django.contrib import admin
from .models import Order, OrderItem, OrderTracking


@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'status', 'total_amount')
    list_display_links = ('id', 'user', 'status')


@admin.register(OrderItem)
class OrderItemAdmin(admin.ModelAdmin):
    list_display = ('id', 'order', 'product', 'quantity')
    list_display_links = ('id', 'order', 'product')


@admin.register(OrderTracking)
class OrderTrackingAdmin(admin.ModelAdmin):
    list_display = ('id', 'order', 'status')
    list_display_links = ('id', 'order')

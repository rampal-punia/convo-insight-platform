from django import forms
from django.forms import inlineformset_factory
from .models import Order, OrderItem


class OrderForm(forms.ModelForm):
    class Meta:
        model = Order
        fields = ['status', 'total_amount']


OrderItemFormSet = inlineformset_factory(
    Order, OrderItem, fields=('product', 'quantity', 'price'), extra=1, can_delete=True
)

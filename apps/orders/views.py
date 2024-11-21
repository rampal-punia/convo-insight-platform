from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.db import transaction
from datetime import timedelta
from django.contrib import messages
from django.core.exceptions import ValidationError
from django.contrib.auth.mixins import LoginRequiredMixin
from django.utils import timezone

from .models import Order, OrderTracking
from .forms import OrderForm, OrderItemFormSet


class OrderListView(LoginRequiredMixin, ListView):
    model = Order
    template_name = 'orders/order_list.html'
    context_object_name = 'orders'

    def get_queryset(self):
        return Order.objects.filter(user=self.request.user)


class OrderDetailView(LoginRequiredMixin, DetailView):
    model = Order
    template_name = 'orders/order_detail.html'
    context_object_name = 'order'


class OrderCreateView(LoginRequiredMixin, CreateView):
    model = Order
    form_class = OrderForm
    template_name = 'orders/order_form.html'
    success_url = reverse_lazy('orders:order_list_url')

    def form_valid(self, form):
        context = self.get_context_data()
        order_items = context['order_items']

        try:
            with transaction.atomic():
                form.instance.user = self.request.user
                form.instance.status = Order.Status.PENDING
                self.object = form.save()

                # Process order items
                if order_items.is_valid():
                    order_items.instance = self.object
                    for form in order_items.forms:
                        if form.is_valid() and not form.cleaned_data.get('DELETE', False):
                            order_item = form.instance
                            order_item.price = order_item.product.price

                    order_items.save()
                else:
                    raise ValidationError("Order items are invalid.")

                # Calculate total amount
                total_amount = sum(
                    item.quantity * item.product.price for item in self.object.items.all())
                self.object.total_amount = total_amount

                # Initialize shipping and tracking information
                self._initialize_shipping_info()

                # Create initial tracking entry
                self._create_initial_tracking()

                # Save with all updates
                self.object.save()

            messages.success(self.request, "Order created successfully")
            return super().form_valid(form)

        except Exception as e:
            messages.error(self.request, f"Error creating order: {str(e)}")
            return super().form_invalid(form)

    def _initialize_shipping_info(self):
        """Initialize shipping-related information based on shipping method"""
        today = timezone.now().date()

        # Set estimated delivery based on shipping method
        delivery_estimates = {
            Order.ShippingMethod.STANDARD: 7,  # 7 days
            Order.ShippingMethod.EXPRESS: 3,   # 3 days
            Order.ShippingMethod.OVERNIGHT: 1,  # 1 day
            Order.ShippingMethod.LOCAL: 2      # 2 days
        }

        # Get shipping method from form or default to standard
        shipping_method = self.object.shipping_method or Order.ShippingMethod.STANDARD

        # Calculate estimated delivery date
        delivery_days = delivery_estimates.get(shipping_method, 7)
        self.object.estimated_delivery = today + timedelta(days=delivery_days)

        # Generate tracking number if not already set
        if not self.object.tracking_number:
            self.object.tracking_number = self._generate_tracking_number()

        # Set carrier based on shipping method
        carriers = {
            Order.ShippingMethod.STANDARD: "Standard Post",
            Order.ShippingMethod.EXPRESS: "Express Courier",
            Order.ShippingMethod.OVERNIGHT: "Priority Express",
            Order.ShippingMethod.LOCAL: "Local Delivery"
        }
        self.object.carrier = carriers.get(shipping_method, "Standard Post")

    def _create_initial_tracking(self):
        """Create the initial tracking entry for the order"""
        description = (
            f"Order #{self.object.id} has been placed successfully. "
            f"Estimated delivery: {self.object.estimated_delivery.strftime('%B %d, %Y')}. "
            f"Shipping via {self.object.get_shipping_method_display()}."
        )

        OrderTracking.objects.create(
            order=self.object,
            status=OrderTracking.TrackingStatus.ORDER_PLACED,
            description=description,
        )

    def _generate_tracking_number(self):
        """Generate a unique tracking number for the order"""
        import random
        import string

        # Generate a random tracking number format: XX-NNNNNN-YY
        # Where X is letter, N is number, Y is letter
        prefix = ''.join(random.choices(string.ascii_uppercase, k=2))
        number = ''.join(random.choices(string.digits, k=6))
        suffix = ''.join(random.choices(string.ascii_uppercase, k=2))

        tracking_number = f"{prefix}-{number}-{suffix}"

        # Ensure uniqueness
        while Order.objects.filter(tracking_number=tracking_number).exists():
            prefix = ''.join(random.choices(string.ascii_uppercase, k=2))
            number = ''.join(random.choices(string.digits, k=6))
            suffix = ''.join(random.choices(string.ascii_uppercase, k=2))
            tracking_number = f"{prefix}-{number}-{suffix}"

        return tracking_number

    def form_invalid(self, form):
        context = self.get_context_data()
        if context['order_items'].is_valid():
            messages.error(
                self.request, "There was an error with the main order form.")
        else:
            messages.error(
                self.request, "There was an error with the order items.")
        return super().form_invalid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.POST:
            context['order_items'] = OrderItemFormSet(
                self.request.POST, instance=self.object)
        else:
            context['order_items'] = OrderItemFormSet(instance=self.object)
        return context


class OrderUpdateView(LoginRequiredMixin, UpdateView):
    model = Order
    form_class = OrderForm
    template_name = 'orders/order_form.html'
    success_url = reverse_lazy('order_list')

    def form_valid(self, form):
        context = self.get_context_data()
        order_items = context['order_items']
        self.object = form.save()
        if order_items.is_valid():
            order_items.instance = self.object
            order_items.save()
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.POST:
            context['order_items'] = OrderItemFormSet(
                self.request.POST, instance=self.object)
        else:
            context['order_items'] = OrderItemFormSet(instance=self.object)
        return context


class OrderDeleteView(LoginRequiredMixin, DeleteView):
    model = Order
    template_name = 'orders/order_confirm_delete.html'
    success_url = reverse_lazy('order_list')

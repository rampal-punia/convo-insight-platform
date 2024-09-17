from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.db import transaction
from django.contrib import messages
from django.core.exceptions import ValidationError
from django.contrib.auth.mixins import LoginRequiredMixin
from .models import Order, OrderItem
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
        with transaction.atomic():
            form.instance.user = self.request.user
            form.instance.status = Order.Status.PENDING
            self.object = form.save()
            if order_items.is_valid():
                order_items.instance = self.object
                order_items.save()
            else:
                raise ValidationError("Order items are invalid.")

            # Calculate total amount
            total_amount = sum(
                item.quantity * item.product.price for item in self.object.items.all())
            self.object.total_amount = total_amount
            self.object.save()

        messages.success(self.request, "Order created successfully")
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.POST:
            context['order_items'] = OrderItemFormSet(
                self.request.POST, instance=self.object)
        else:
            context['order_items'] = OrderItemFormSet(instance=self.object)
        return context

    def form_invalid(self, form):
        context = self.get_context_data()
        if context['order_items'].is_valid():
            messages.error(
                self.request, "There was an error with the main order form.")
        else:
            messages.error(
                self.request, "There was an error with the order items.")
        return super().form_invalid(form)


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

from django.urls import path
from . import views

app_name = 'orders'

urlpatterns = [
    path('', views.OrderListView.as_view(), name='order_list_url'),
    path('<int:pk>/', views.OrderDetailView.as_view(), name='order_detail_url'),
    path('create/', views.OrderCreateView.as_view(), name='order_create_url'),
    path('<int:pk>/update/', views.OrderUpdateView.as_view(),
         name='order_update_url'),
    path('<int:pk>/delete/', views.OrderDeleteView.as_view(),
         name='order_delete_url'),
]

from django.urls import path
from . import views

app_name = 'general_assistant'

urlpatterns = [
    path("", views.GeneralConversationListView.as_view(),
         name='generalchat_list_url'),
    path('chat/', views.GeneralConversationDetailView.as_view(),
         name='new_generalchat_url'),
    path('chat/<uuid:pk>/', views.GeneralConversationDetailView.as_view(),
         name='generalchat_detail_url'),
    path("<uuid:pk>/delete/", views.GeneralConversationDeleteView.as_view(),
         name='delete_url'),
    path("new/", views.GeneralConversationView.as_view(),
         name='new_general_chat_url'),
]

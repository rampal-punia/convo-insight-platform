# config/urls.py

from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from apps.accounts import views

urlpatterns = [
    path('accounts/login/', views.CustomLoginView.as_view(), name='account_login'),
    path('accounts/signup/', views.CustomSignupView.as_view(), name='account_signup'),
    path('accounts/password/reset/', views.CustomPasswordResetView.as_view(),
         name='account_reset_password'),
    path('accounts/profile/', views.CustomProfileView.as_view(),
         name='account_profile'),
    path('logout/', views.CustomLogoutView.as_view(), name='account_logout'),
    path('accounts/', include('allauth.urls')),
    path('admin/', admin.site.urls),
    path('', include('dashboard.urls'), name='dashboard'),
    path('convochat/', include('convochat.urls'), name='convochat'),
    path('general_assistant/', include('general_assistant.urls'),
         name='general_assistant'),
    path('orders/', include('orders.urls'),
         name='orders'),
    path('playground/', include('playground.urls'),
         name='playground'),
    path('support_agent/', include('support_agent.urls'),
         name='support_agent'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL,
                          document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)


if settings.DEBUG and 'debug_toolbar' in settings.INSTALLED_APPS:
    import debug_toolbar
    urlpatterns += [
        path('__debug__/', include(debug_toolbar.urls)),
    ]

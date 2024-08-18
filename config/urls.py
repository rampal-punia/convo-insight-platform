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
    # path('accounts/logout/', CustomLogoutView.as_view(), name='account_logout'),
    path('accounts/', include('allauth.urls')),
    path('admin/', admin.site.urls),
    path('dashboard/', include('dashboard.urls'), name='dashboard'),
    path('convochat/', include('convochat.urls'), name='convochat'),
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

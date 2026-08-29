"""
=============================================================================
Project Vani: Django Root URL Configuration
=============================================================================
Routes all root /api/ traffic to the API application endpoints.
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),
]

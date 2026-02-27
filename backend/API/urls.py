from django.urls import path
from .views import ImageUploadView, ComparisonView, health_check

urlpatterns = [
    path('upload/', ImageUploadView.as_view(), name='image-upload'),
    path('compare/', ComparisonView.as_view(), name='comparison'),
    path('health/', health_check, name='health-check'),
]
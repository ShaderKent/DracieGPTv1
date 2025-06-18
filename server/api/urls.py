from django.urls import path
from .views import get_query

urlpatterns = [
    path('query/', get_query, name="get_query"),
]
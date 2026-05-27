"""Pagination classes shared across the v1 API."""
from rest_framework.pagination import PageNumberPagination


class StandardResultsSetPagination(PageNumberPagination):
    """Default page-number pagination tuned for list dashboards."""

    page_size = 25
    page_size_query_param = "page_size"
    max_page_size = 200

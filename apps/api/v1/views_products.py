"""ViewSets for the Products domain."""
from django.db.models import Count
from django_filters.rest_framework import DjangoFilterBackend
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import filters, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from products.models import Category, Product

from ..pagination import StandardResultsSetPagination
from .serializers_products import CategorySerializer, ProductSerializer


class CategoryViewSet(viewsets.ModelViewSet):
    """CRUD for product categories."""

    queryset = Category.objects.all().order_by("name")
    serializer_class = CategorySerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["name", "description"]
    ordering_fields = ["name", "id"]

    def get_queryset(self):
        return super().get_queryset().annotate(product_count=Count("product"))

    @extend_schema(
        summary="List products belonging to a category",
        responses=ProductSerializer(many=True),
    )
    @action(detail=True, methods=["get"], url_path="products", url_name="products")
    def products(self, request, pk=None):
        category = self.get_object()
        qs = Product.objects.filter(category=category).order_by("-created")
        page = self.paginate_queryset(qs)
        serializer = ProductSerializer(page or qs, many=True)
        if page is not None:
            return self.get_paginated_response(serializer.data)
        return Response(serializer.data)


class ProductViewSet(viewsets.ModelViewSet):
    """CRUD for products with search, filter and stock helpers."""

    queryset = Product.objects.select_related("category").all().order_by("-created")
    serializer_class = ProductSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["category", "stock"]
    search_fields = ["name", "description"]
    ordering_fields = ["price", "stock", "created", "modified", "name"]

    @extend_schema(
        summary="List products that are currently in stock",
        responses=ProductSerializer(many=True),
    )
    @action(detail=False, methods=["get"], url_path="in-stock", url_name="in-stock")
    def in_stock(self, request):
        qs = self.get_queryset().filter(stock__gt=0)
        page = self.paginate_queryset(qs)
        serializer = self.get_serializer(page or qs, many=True)
        if page is not None:
            return self.get_paginated_response(serializer.data)
        return Response(serializer.data)

    @extend_schema(
        summary="List low-stock products",
        parameters=[
            OpenApiParameter(
                name="threshold",
                type=int,
                location=OpenApiParameter.QUERY,
                description="Stock-level threshold (default 5).",
                required=False,
            )
        ],
        responses=ProductSerializer(many=True),
    )
    @action(detail=False, methods=["get"], url_path="low-stock", url_name="low-stock")
    def low_stock(self, request):
        try:
            threshold = int(request.query_params.get("threshold", 5))
        except (TypeError, ValueError):
            threshold = 5
        qs = self.get_queryset().filter(stock__lte=threshold)
        page = self.paginate_queryset(qs)
        serializer = self.get_serializer(page or qs, many=True)
        if page is not None:
            return self.get_paginated_response(serializer.data)
        return Response(serializer.data)

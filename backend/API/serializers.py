from rest_framework import serializers

# Simple serializers for file-based operations (no database models)

class ImageUploadSerializer(serializers.Serializer):
    """Serializer for single image upload and processing"""
    image = serializers.ImageField()
    algorithm = serializers.CharField(default='espcn')
    scale = serializers.CharField(default='2')

class ComparisonSerializer(serializers.Serializer):
    """Serializer for algorithm comparison"""
    image = serializers.ImageField()
    algorithms = serializers.ListField(child=serializers.CharField())
    scale = serializers.CharField(default='2')
    parameters = serializers.DictField(required=False, default=dict)

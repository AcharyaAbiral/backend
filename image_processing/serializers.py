from rest_framework import serializers


class PredictionImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()


class FaceRegistrationImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()
    name = serializers.CharField()
    # serializers.Fi

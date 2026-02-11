# main/models.py
from django.db import models
from django.utils.text import slugify

class BirdSpecies(models.Model):
    # Step 1: Basic info
    species_name = models.CharField(max_length=200)
    images = models.ManyToManyField('BirdImage', blank=True)
    audios = models.ManyToManyField('BirdAudio', blank=True)

    # Step 2: Details
    scientific_name = models.CharField(max_length=200, blank=True)
    family_name = models.CharField(max_length=200, blank=True)
    description = models.TextField(blank=True)
    iucn_category = models.CharField(max_length=50, blank=True)

    # Flag to indicate manual entry
    is_manual = models.BooleanField(default=False)

    # Slug field for URL
    slug = models.SlugField(max_length=250, unique=True, blank=True, null=True)

    def __str__(self):
        return self.species_name

    # Automatically generate slug if not provided
    def save(self, *args, **kwargs):
        if not self.slug:
            base_slug = slugify(self.species_name)
            slug = base_slug
            counter = 1
            # Ensure slug uniqueness
            while BirdSpecies.objects.filter(slug=slug).exists():
                slug = f"{base_slug}-{counter}"
                counter += 1
            self.slug = slug
        super().save(*args, **kwargs)


class BirdLocation(models.Model):
    """
    Separate model to store multiple locations for a bird.
    This allows multiple points on the map without overwriting old data.
    """
    bird = models.ForeignKey(BirdSpecies, on_delete=models.CASCADE, related_name='locations')
    district = models.CharField(max_length=100, blank=True, null=True)
    location_name = models.CharField(max_length=200, blank=True, null=True)
    latitude = models.DecimalField(max_digits=8, decimal_places=5)
    longitude = models.DecimalField(max_digits=8, decimal_places=5)

    def __str__(self):
        return f"{self.bird.species_name} - {self.location_name} ({self.latitude}, {self.longitude})"


class BirdImage(models.Model):
    image = models.ImageField(upload_to='bird_images/')

    def __str__(self):
        return self.image.name


class BirdAudio(models.Model):
    audio = models.FileField(upload_to='bird_audios/')

    def __str__(self):
        return self.audio.name

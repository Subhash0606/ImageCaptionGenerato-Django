from django.db import models

# Create your models here.

from django.db import models

class ImageUpload(models.Model):
    image = models.ImageField(upload_to='static/')

    def get_image_url(self):
        return self.image.url

# config/models.py

import logging

from django.db import models
from django.utils.translation import gettext_lazy as _

logger = logging.getLogger('config')


class CreationModificationDateBase(models.Model):
    """
    Abstract base class with date and time field for creation and modification 
    """

    created = models.DateTimeField(
        _("Creation Date and Time"),
        auto_now_add=True,
    )

    modified = models.DateTimeField(
        _("Modification Date and Time"),
        auto_now=True,
    )

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        logger.info("save() from CreationModificationDateBase called")
    save.alters_data = True

    def delete(self, *args, **kwargs):
        super().delete(*args, **kwargs)
        logger.info("delete() from CreationModificationDateBase called")

    def test(self):
        logger.info("test() from CreationModificationDateBase called")

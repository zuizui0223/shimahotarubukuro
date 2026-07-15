# -*- coding: utf-8 -*-
"""Separate added red/green review strokes from natural specimen colours."""
from __future__ import annotations

import cv2
import numpy as np


def annotation_masks(annotated: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return red boundary and green organ strokes added to an annotated preview.

    The raw scan is resized to the preview. Natural red nectar-guide spots and
    greenish specimen tissue occur in both images and are rejected unless colour
    dominance increases strongly in the reviewed image.
    """
    if annotated.ndim != 3 or raw.ndim != 3:
        raise ValueError("annotated and raw images must be BGR colour images")
    reference = cv2.resize(raw, (annotated.shape[1], annotated.shape[0]), interpolation=cv2.INTER_AREA)

    ann = annotated.astype(np.int16)
    ref = reference.astype(np.int16)
    ab, ag, ar = cv2.split(ann)
    rb, rg, rr = cv2.split(ref)

    hsv = cv2.cvtColor(annotated, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    vivid = (sat >= 105) & (val >= 80)

    red_dom = ar - np.maximum(ag, ab)
    ref_red_dom = rr - np.maximum(rg, rb)
    green_dom = ag - np.maximum(ar, ab)
    ref_green_dom = rg - np.maximum(rr, rb)
    colour_delta = np.sqrt(np.sum((ann.astype(np.float32) - ref.astype(np.float32)) ** 2, axis=2))

    red_hue = (hue <= 12) | (hue >= 168)
    green_hue = (hue >= 35) & (hue <= 95)
    red = vivid & red_hue & (red_dom >= 55) & ((red_dom - ref_red_dom >= 35) | (colour_delta >= 70))
    green = vivid & green_hue & (green_dom >= 45) & ((green_dom - ref_green_dom >= 30) | (colour_delta >= 65))

    kernel = np.ones((3, 3), np.uint8)
    red = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return red, green

import os

# The bundled models are Keras 2 JSON, which Keras 3 cannot deserialise, so
# tf-keras is used. tensorflow.keras resolves when it is first imported, so
# this has to be set before that happens.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

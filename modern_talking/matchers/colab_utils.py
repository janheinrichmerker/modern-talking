from os import environ

from tensorflow import distribute, config, tpu


def is_running_on_colab():
    return 'COLAB_GPU' in environ


def setup_colab_tpu():
    """
    Setup TPUs for usage in Google Colab, only if running on Colab.
    """
    if not is_running_on_colab():
        return
        # Special resolver for Google Colaboratory.
    resolver = distribute.cluster_resolver.TPUClusterResolver(tpu='')
    config.experimental_connect_to_cluster(resolver)
    tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", config.list_logical_devices("TPU"))

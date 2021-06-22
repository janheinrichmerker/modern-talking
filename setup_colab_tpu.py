from tensorflow import distribute, config, tpu
# Special resolver for Google Colaboratory.
resolver = distribute.cluster_resolver.TPUClusterResolver(tpu='')
config.experimental_connect_to_cluster(resolver)
tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", config.list_logical_devices("TPU"))

from tensorflow.config.experimental import set_virtual_device_configuration, \
VirtualDeviceConfiguration, list_physical_devices, list_logical_devices

def use_gpu(gpu_mem_gb):

    gpus = list_physical_devices('GPU')
    
    if gpus:
    
        try:
            set_virtual_device_configuration(
                gpus[0],
                [VirtualDeviceConfiguration(memory_limit=(1024*gpu_mem_gb))]
            )
            logical_gpus = list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as error:
           
            print(error)

from .sparkvsr_native.nodes import SparkVSRNativeExtension
WEB_DIRECTORY = "./web"

  
async def comfy_entrypoint() -> SparkVSRNativeExtension:
    return SparkVSRNativeExtension()

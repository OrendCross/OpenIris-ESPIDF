idf_component_register(SRCS "JpegLsEncoder/jpegls_encoder.cpp"
  INCLUDE_DIRS "JpegLsEncoder"
)

# Use -O3 for the encoder hot path (default is -O2)
target_compile_options(${COMPONENT_LIB} PRIVATE -O3)

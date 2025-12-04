idf_component_register(
    SRCS   "model2_data_tts.c" "main.cpp" 
    INCLUDE_DIRS "."
    REQUIRES freertos esp_timer
)
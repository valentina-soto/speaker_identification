#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_mac.h"
#include "esp_timer.h"  

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

//#include "quantized_model2.h" 
//#include "model_data_tts.h" 
//#include "model_data_espec.h" 
//#include "model_data_espec_tts.h"
//#include "model_data_mfcc.h"
//#include "model_data_mfcc_tts.h"
#include "model2_data_tts.h"

extern const float test_data[100][256];
extern const int clases[100];
extern const unsigned char modelo[];
extern const unsigned int modelo_len;

constexpr int kTensorArenaSize = 120 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

extern "C" void app_main(void)
{
    printf("funcionando\n");

    const tflite::Model* model = tflite::GetModel(modelo);

    // Resolver de operaciones
    tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, nullptr);

    interpreter.AllocateTensors();

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    while (true) {

        printf("\n=== NUEVA RONDA ===\n");
        printf("Esperando CSV...\n");

        int count = 0;
        int64_t suma_tiempos = 0;   // para promedio

        for (int i = 0; i < 100; i++) {
            printf("\nProcesando Caso #%d:\n", i);

            // Copiar datos al tensor de entrada
            memcpy(input->data.f, test_data[i], sizeof(float) * 256);

            // ---------- MEDIR INFERENCIA ----------
            int64_t t0 = esp_timer_get_time();   // tiempo inicial (us)
            TfLiteStatus status = interpreter.Invoke();
            int64_t t1 = esp_timer_get_time();   // tiempo final (us)
            int64_t duracion = t1 - t0;
            suma_tiempos += duracion;

            printf("Tiempo de inferencia: %lld us\n", duracion);
            // --------------------------------------

            if (status == kTfLiteOk) {
                float* resultados = output->data.f;
                float max_prob = -1.0f;
                int clase_ganadora = -1;

                printf("Probabilidades: [ ");
                for (int j = 0; j < 4; j++) {
                    float prob = resultados[j];
                    printf("%.2f ", prob);
                    if (prob > max_prob) {
                        max_prob = prob;
                        clase_ganadora = j;
                    }
                }
                printf("]\n");

                printf("Clase %d (%.1f%%), clase esperada: %d\n",
                       clase_ganadora, max_prob * 100, clases[i]);

                if (clase_ganadora == clases[i]) {
                    count++;
                }

            } else {
                printf("ERROR: Invoke falló en caso %d\n", i);
            }

            vTaskDelay(200 / portTICK_PERIOD_MS);
        }

        float promedio = (float)suma_tiempos / 100.0f;

        printf("\n Resultados tras 100 casos: \n");
        printf("Aciertos: %d/100\n", count);
        printf("Tiempo promedio de inferencia: %.2f ms\n", promedio/1000.0f);
        printf("\n");

        vTaskDelay(10000 / portTICK_PERIOD_MS);
    }
}

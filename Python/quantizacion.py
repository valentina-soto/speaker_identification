import tensorflow as tf
import glob

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("model_mfcc_tts.tflite","wb").write(tflite_model)

def representative_dataset_gen():
    csv_files = glob.glob("/content/drive/MyDrive/Certamen_Softcomputing/dataset_csv_mfcc_tts/train/*.csv")[:100]  # 100 muestras
    for f in csv_files:
        df = pd.read_csv(f)
        pixels = df[[c for c in df.columns if c.startswith("p")]].values.astype(np.float32)
        #pixels = pixels / 255.0
        yield [pixels]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                       tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_quant = converter.convert()
open("quantized_model_mfcc_tts.tflite","wb").write(tflite_quant)
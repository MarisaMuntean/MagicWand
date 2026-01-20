#include "Arduino.h"
#include <TensorFlowLite_ESP32.h>
#include "model.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <Wire.h>
#include <MPU6050.h>

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;

const int num_classes = 4;  // Number of classes (e.g., if recording 4 actions, set to 5, etc.)
const int input_dim = 2;    // Sensor data dimensions (e.g., if recording X and Z axes, set to 2)

constexpr int kTensorArenaSize = 50 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}

MPU6050 mpu;

void resetState(); 
void get_kalman_mpu_data(int i, float* input);
void processGesture(float* output);

// Sampling frequency (Hz)
const int freq = 100;
const int second = 2;

// Gravity components
float gravity_x;
float gravity_y;
float gravity_z;

// Angular velocity converted to X/Y axes
float roll_v, pitch_v;

// Last update time
unsigned long prevTime;

// Kalman filter states: predicted, observed, and optimal estimate
float gyro_roll, gyro_pitch;  // Gyroscope-integrated angles (predicted state)
float acc_roll, acc_pitch;    // Accelerometer-measured angles (observed state)
float k_roll, k_pitch;        // Kalman-filtered optimal angles (estimated state)

// Error covariance matrix P
float e_P[2][2];  // Also serves as prior and updated P

// Kalman gain matrix K (2x2)
float k_k[2][2];

const int buttonPin = 41;  // Button pin (change as needed)
const int ledPin = 42;    // LED pin (optional)
int buttonState;          // Current button state
int lastButtonState = HIGH; // Previous button state

unsigned long lastDebounceTime = 0; // Last debounce time
unsigned long debounceDelay = 10;   // Debounce delay (ms)

void setup() {
  Serial.begin(115200);

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
      "Model version %d does not match supported version %d.",
      model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::AllOpsResolver resolver;
  resolver.AddConv2D();
  resolver.AddRelu();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddTranspose();
  resolver.AddExpandDims();

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  interpreter->AllocateTensors();
  model_input = interpreter->input(0);

  Wire.begin(4, 5);
  mpu.initialize();

  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed");
    while (1);
  }

  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(ledPin, OUTPUT);
  
  pinMode(37, OUTPUT);  // Updated pin 21 → 37
  pinMode(36, OUTPUT);  // Updated pin 20 → 36
  pinMode(35, OUTPUT);  // Updated pin 19 → 35
  digitalWrite(ledPin, LOW);
  resetState();
}

void loop() {
  int reading = digitalRead(buttonPin);

  // Detect button state change
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }

  // Apply debounce
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != buttonState) {
      buttonState = reading;

      // Trigger on button release
      if (buttonState == HIGH) {
        resetState();
        digitalWrite(ledPin, HIGH);

        // Collect sensor data
        for (int i = 0; i < freq * second; i++) {
          get_kalman_mpu_data(i, model_input->data.f);
        }

        // Run inference
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
          error_reporter->Report("Invoke failed");
          return;
        }

        processGesture(interpreter->output(0)->data.f);
        delay(100);
        digitalWrite(ledPin, LOW);
      }
    }
  }

  lastButtonState = reading;
}

void resetState() {
  // Read accelerometer data
  int16_t ax, ay, az;
  mpu.getAcceleration(&ax, &ay, &az);

  // Convert to g-values
  float Ax = ax / 16384.0;
  float Ay = ay / 16384.0;
  float Az = az / 16384.0;

  // Calculate initial pitch/roll
  k_pitch = -atan2(Ax, sqrt(Ay * Ay + Az * Az));
  k_roll = atan2(Ay, Az);

  // Initialize covariance matrix
  e_P[0][0] = 1;
  e_P[0][1] = 0;
  e_P[1][0] = 0;
  e_P[1][1] = 1;

  // Initialize Kalman gain
  k_k[0][0] = 0;
  k_k[0][1] = 0;
  k_k[1][0] = 0;
  k_k[1][1] = 0;

  prevTime = millis();
}

void get_kalman_mpu_data(int i, float* input) {
  // Calculate time delta
  unsigned long currentTime = millis();
  float dt = (currentTime - prevTime) / 1000.0;
  prevTime = currentTime;

  // Read IMU data
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Convert accelerometer data to g-values
  float Ax = ax / 16384.0;
  float Ay = ay / 16384.0;
  float Az = az / 16384.0;
  float Ox, Oy, Oz;

  // Convert gyro data to rad/s
  float Gx = gx / 131.0 / 180 * PI;
  float Gy = gy / 131.0 / 180 * PI;
  float Gz = gz / 131.0 / 180 * PI;

  // Step 1: Predict state
  roll_v = Gx + ((sin(k_pitch) * sin(k_roll)) / cos(k_pitch)) * Gy + ((sin(k_pitch) * cos(k_roll)) / cos(k_pitch)) * Gz;
  pitch_v = cos(k_roll) * Gy - sin(k_roll) * Gz;
  gyro_roll = k_roll + dt * roll_v;
  gyro_pitch = k_pitch + dt * pitch_v;

  // Step 2: Predict covariance
  e_P[0][0] += 0.0025;  // Q matrix diagonal
  e_P[1][1] += 0.0025;

  // Step 3: Update Kalman gain
  k_k[0][0] = e_P[0][0] / (e_P[0][0] + 0.3);
  k_k[1][1] = e_P[1][1] / (e_P[1][1] + 0.3);

  // Step 4: Update estimate
  acc_roll = atan2(Ay, Az);
  acc_pitch = -atan2(Ax, sqrt(Ay * Ay + Az * Az));
  k_roll = gyro_roll + k_k[0][0] * (acc_roll - gyro_roll);
  k_pitch = gyro_pitch + k_k[1][1] * (acc_pitch - gyro_pitch);

  // Step 5: Update covariance
  e_P[0][0] = (1 - k_k[0][0]) * e_P[0][0];
  e_P[1][1] = (1 - k_k[1][1]) * e_P[1][1];

  // Calculate gravity components
  gravity_x = -sin(k_pitch);
  gravity_y = sin(k_roll) * cos(k_pitch);
  gravity_z = cos(k_roll) * cos(k_pitch);

  // Remove gravity
  Ax -= gravity_x;
  Ay -= gravity_y;
  Az -= gravity_z;

  // Transform to global coordinates
  Ox = cos(k_pitch) * Ax + sin(k_pitch) * sin(k_roll) * Ay + sin(k_pitch) * cos(k_roll) * Az;
  Oy = cos(k_roll) * Ay - sin(k_roll) * Az;
  Oz = -sin(k_pitch) * Ax + cos(k_pitch) * sin(k_roll) * Ay + cos(k_pitch) * cos(k_roll) * Az;

  // Store processed data
  input[i * input_dim] = Oy;
  input[i * input_dim + 1] = Oz;

  delay(1000 / freq);  // Maintain sampling rate
}

void processGesture(float* output) {
  int max_index = 0;
  float max_value = output[0];

  for (int i = 1; i < num_classes; i++) {
    if (output[i] > max_value) {
      max_value = output[i];
      max_index = i;
    }
  }

  // Action handling
  switch (max_index) {
    case 0:  // Rock
      Serial.println("rock");
      digitalWrite(37, HIGH);
      delay(200);
      digitalWrite(37, LOW);
      break;
    case 1:  // Scissors
      Serial.println("scissors");
      digitalWrite(36, HIGH);
      delay(200);
      digitalWrite(36, LOW);
      break;
    case 2:  // Paper
      Serial.println("paper");
      digitalWrite(35, HIGH);
      delay(200);
      digitalWrite(35, LOW);
      break;
  }
}
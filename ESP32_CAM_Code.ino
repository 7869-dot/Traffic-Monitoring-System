#include "WiFi.h"
#include "esp_camera.h"
#include "HTTPClient.h"

// ------------------- WiFi CONFIG -------------------
const char* WIFI_SSID     = "HUAWEI-2.4G-P39g";
const char* WIFI_PASSWORD = "e5c65VE5";

// FastAPI server URL (HTTP)
// Update this to match your FastAPI server IP address
String serverUrl = "http://192.168.1.100:8000/upload_frame";  

// ------------------- CAMERA PINS (AI THINKER) -------------------
#define CAMERA_MODEL_AI_THINKER

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ------------------- CONFIGURATION -------------------
const unsigned long FRAME_INTERVAL_MS = 500;  // Send frame every 500ms (2 FPS)
const int MAX_WIFI_RETRIES = 5;               // Max WiFi reconnection attempts
const int HTTP_TIMEOUT_MS = 10000;            // HTTP timeout (10 seconds)
const int MAX_HTTP_RETRIES = 3;               // Max HTTP retry attempts

// ------------------- WIFI CONNECT -------------------
bool connectToWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int maxTries = 30; // ~15 seconds
  int tries = 0;

  while (WiFi.status() != WL_CONNECTED && tries < maxTries) {
    delay(500);
    Serial.print(".");
    tries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal strength (RSSI): ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
    return true;
  } else {
    Serial.println("\nFailed to connect to WiFi.");
    return false;
  }
}

// ------------------- CAMERA SETUP -------------------
bool setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Frame size + quality
  config.frame_size   = FRAMESIZE_QVGA;  // 320x240
  config.jpeg_quality = 12;             // 0–63, lower = better quality
  config.fb_count     = 1;

  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return false;
  }

  Serial.println("Camera initialized successfully.");
  return true;
}

// ------------------- SEND FRAME TO SERVER -------------------
bool sendFrameToServer(camera_fb_t *fb) {
  if (!fb || fb->len == 0) {
    Serial.println("Error: Invalid frame buffer");
    return false;
  }

  HTTPClient http;
  WiFiClient client;

  // Configure HTTP client
  http.setTimeout(HTTP_TIMEOUT_MS);
  http.begin(client, serverUrl);

  // Set headers
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-Camera-Id", "esp32_cam_1");

  // Send POST request with retry logic
  int httpResponseCode = -1;
  int retries = 0;

  while (retries < MAX_HTTP_RETRIES && httpResponseCode <= 0) {
    if (retries > 0) {
      Serial.printf("Retrying HTTP request (attempt %d/%d)...\n", retries + 1, MAX_HTTP_RETRIES);
      delay(1000 * retries); // Exponential backoff
    }

    httpResponseCode = http.POST(fb->buf, fb->len);

    if (httpResponseCode > 0) {
      Serial.printf("HTTP Response code: %d\n", httpResponseCode);
      
      if (httpResponseCode == 200) {
        String response = http.getString();
        Serial.print("Server response: ");
        Serial.println(response);
        http.end();
        return true;
      } else {
        Serial.printf("Server returned error code: %d\n", httpResponseCode);
        String response = http.getString();
        Serial.print("Error response: ");
        Serial.println(response);
      }
    } else {
      Serial.printf("HTTP POST failed: %s\n", http.errorToString(httpResponseCode).c_str());
    }

    retries++;
  }

  http.end();
  return false;
}

// ------------------- ARDUINO SETUP -------------------
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n========================================");
  Serial.println("ESP32-CAM → FastAPI (HTTP POST)");
  Serial.println("========================================\n");

  // Setup camera
  if (!setupCamera()) {
    Serial.println("FATAL: Camera initialization failed. Restarting...");
    delay(5000);
    ESP.restart();
  }

  // Connect to WiFi
  int wifiRetries = 0;
  while (!connectToWiFi() && wifiRetries < MAX_WIFI_RETRIES) {
    wifiRetries++;
    Serial.printf("WiFi connection attempt %d/%d failed. Retrying...\n", wifiRetries, MAX_WIFI_RETRIES);
    delay(2000);
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("FATAL: Could not connect to WiFi. Restarting...");
    delay(5000);
    ESP.restart();
  }

  Serial.println("\nSetup complete. Starting main loop...\n");
}

// ------------------- MAIN LOOP -------------------
void loop() {
  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected. Attempting to reconnect...");
    if (!connectToWiFi()) {
      Serial.println("WiFi reconnection failed. Waiting before retry...");
      delay(5000);
      return;
    }
  }

  // Capture frame
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    delay(500);
    return;
  }

  Serial.printf("Captured frame: %d bytes\n", fb->len);

  // Send frame to server
  bool success = sendFrameToServer(fb);

  if (success) {
    Serial.println("Frame sent successfully");
  } else {
    Serial.println("Failed to send frame to server");
  }

  // Return frame buffer to driver
  esp_camera_fb_return(fb);

  // Wait before capturing next frame
  delay(FRAME_INTERVAL_MS);
}


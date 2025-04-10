#include "LSM6DS3.h"
#include <bluefruit.h>
// Comment out magnetometer include
// #include <QMC5883LCompass.h>
#include <MadgwickAHRS.h>
#include <Wire.h>

// Data Transfer Speed
const int sendEvery = 100;

// Comment out magnetometer instance
// QMC5883LCompass myMag;

//Madgwick Filter Instance
Madgwick filter;

// IMU Instance
LSM6DS3 myIMU(I2C_MODE, 0x6A);

// Multiplexer control pin
const int muxS0 = D8;

// Variables for timestamp tracking
uint32_t initTimestamp = 0;
uint32_t receivedTimestamp = 0;
uint32_t deviceTimestamp = 0;
uint32_t startTime = 0;
bool hasReceivedTimestamp = false;

// Variables for flexsensor pins
const int flexSensor1 = A0;
const int flexSensor2 = A1;
const int flexSensor3 = A2;
const int flexSensor4_5 = A3;

// Variables for complementary filter
float compRoll = 0;
float compPitch = 0;
float alpha = 0.96; // Complementary filter coefficient

// Variables for Kalman filter
float kalmanRoll = 0;
float kalmanPitch = 0;
float kalmanUncertaintyRoll = 2*2;
float kalmanUncertaintyPitch = 2*2;

// Variables for Flex Sensor Kalman filters
struct KalmanState {
    float state;
    float uncertainty;
};

KalmanState flexKalman[5] = {
    {0, 2*2}, // Flex1
    {0, 2*2}, // Flex2
    {0, 2*2}, // Flex3
    {0, 2*2}, // Flex4
    {0, 2*2}  // Flex5
};

// Previous flex sensor readings for velocity estimation
float prevFlexValues[5] = {0, 0, 0, 0, 0};

// Time tracking variables
unsigned long previousTime = 0;
float deltaTime = 0;

// Service UUID for Left Glove
const uint8_t RANDOM_SERVICE_UUID[] = {
    0x78, 0x56, 0x34, 0x12, 0x00, 0x00, 0xF7, 0x1E, 
    0x80, 0x00, 0x00, 0x80, 0x5F, 0x9B, 0x34, 0xFB
};

// Service UUID for Right Glove
// const uint8_t RANDOM_SERVICE_UUID[] = {
//     0x78, 0x56, 0x34, 0x12, 0x00, 0x00, 0x7E, 0x41, 
//     0x80, 0x00, 0x00, 0x80, 0x5F, 0x9B, 0x34, 0xFB
// };

const uint8_t RANDOM_CHARACTERISTIC_UUID[] = {
    0x78, 0x56, 0x00, 0x00, 0x10, 0x00, 
    0x80, 0x00, 0x00, 0x80, 0x5F, 0x9B, 0x34, 0xFB
};

// Bluetooth Service and Characteristics
BLEService dataService = BLEService(RANDOM_SERVICE_UUID); // Standard Service UUID or use custom
BLECharacteristic randomCharacteristic = BLECharacteristic(RANDOM_CHARACTERISTIC_UUID); // Standard Characteristic UUID or use custom

// Function to calculate angles from accelerometer data
void calculateAccelAngles(float ax, float ay, float az, float& roll, float& pitch) {
    roll = atan2(ay, sqrt(ax*ax + az*az)) * 180.0/M_PI;
    pitch = atan2(-ax, sqrt(ay*ay + az*az)) * 180.0/M_PI;
}

// Kalman filter implementation
void kalmanFilter(float& kalmanState, float& kalmanUncertainty, float gyroRate, float accelAngle) {
    // Predict
    kalmanState = kalmanState + deltaTime * gyroRate;
    kalmanUncertainty = kalmanUncertainty + deltaTime * deltaTime * 4 * 4; // Process noise
    
    // Update
    float kalmanGain = kalmanUncertainty / (kalmanUncertainty + 3 * 3); // Measurement noise
    kalmanState = kalmanState + kalmanGain * (accelAngle - kalmanState);
    kalmanUncertainty = (1 - kalmanGain) * kalmanUncertainty;
}

// Kalman filter implementation for flex sensors
float kalmanFilterFlex(KalmanState& ks, float measurement, float& prevValue) {
    // Calculate rate of change (velocity)
    float velocity = (measurement - prevValue) / deltaTime;
    prevValue = measurement;
    
    // Predict
    float predictedState = ks.state + velocity * deltaTime;
    float predictedUncertainty = ks.uncertainty + deltaTime * deltaTime * 2 * 2; // Process noise was 1
    
    // Update
    float kalmanGain = predictedUncertainty / (predictedUncertainty + 2 * 2); // Measurement noise was 25
    ks.state = predictedState + kalmanGain * (measurement - predictedState);
    ks.uncertainty = (1 - kalmanGain) * predictedUncertainty;
    
    return ks.state;
}

void switchToFlexSensor4() {
  digitalWrite(muxS0, LOW);
}

void switchToFlexSensor5() {
  digitalWrite(muxS0, HIGH);
}

// Callback function for incoming data
void onDataReceived(uint16_t conn_handle, BLECharacteristic* characteristic, uint8_t* data, uint16_t len) {
  // Read data sent
  if (len == sizeof(uint32_t)*2) {
    // Read sent timestamps
    memcpy(&receivedTimestamp, data, 4);
    memcpy(&initTimestamp, data + 4, 4);

    startTime = millis();
    hasReceivedTimestamp = true; 
  }
  // Comment out the calibration section since we're removing the magnetometer
  /*
  else if (len == sizeof(uint32_t)*6) {
    // Read sent Calibration data
    Serial.println("Received Calibration Settings");
    uint32_t cal1, cal2, cal3, cal4, cal5, cal6;

    memcpy(&cal1, data, 4);
    memcpy(&cal2, data + 4, 4);
    memcpy(&cal3, data + 8, 4);
    memcpy(&cal4, data + 16, 4);
    memcpy(&cal5, data + 20, 4);
    memcpy(&cal6, data + 24, 4);

    myMag.setCalibration(cal1, cal2, cal3, cal4, cal5, cal6);
  }
  */
}

void updateTimestamp() {
  // Update the timestamp on the device
  unsigned long currentTime = millis();
  deviceTimestamp = (currentTime - startTime) + receivedTimestamp;
}


void setup() {
  analogReadResolution(12);
  previousTime = millis();
  Serial.begin(115200);

  if (myIMU.begin() != 0) {
        Serial.println("IMU initialization failed!");
    } else {
        Serial.println("IMU initialized.");
    }

  // Setup Pin for multiplexer
  pinMode(muxS0, OUTPUT);

  // Configure the MTU for Bluetooth
  Bluefruit.configPrphBandwidth(BANDWIDTH_HIGH);
  Bluefruit.configCentralBandwidth(BANDWIDTH_HIGH);

  // Initialize Bluefruit
  if (!Bluefruit.begin()) {
    Serial.println("Bluefruit initialization failed!");
    while (1);
  }

  // Set up the BLE service
  Bluefruit.setName("SignLanguageGlove");
  dataService.begin();

  // Set up the characteristic
  randomCharacteristic.setProperties(CHR_PROPS_WRITE | CHR_PROPS_WRITE_WO_RESP | CHR_PROPS_READ | CHR_PROPS_NOTIFY);
  randomCharacteristic.setPermission(SECMODE_OPEN, SECMODE_OPEN);
  randomCharacteristic.setFixedLen(60);
  randomCharacteristic.setWriteCallback(onDataReceived);
  randomCharacteristic.begin();

  // Start advertising
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(dataService);
  Bluefruit.Advertising.start();
  Bluefruit.setName("SignLanguageGlove");

  // Comment out magnetometer initialization
  // myMag.init();

  // Initialize Madgwick filter - switch to 6DOF mode (no magnetometer)
  filter.begin(100); // Sample rate of 100 Hz
}

void loop() {
  while (!Bluefruit.connected()) {
    hasReceivedTimestamp = false;
    Serial.println("Not connected");
    delay(1000);
  }

  Serial.println("Device connected. Awaiting timestamp...");

  while (Bluefruit.connected() && hasReceivedTimestamp) {

    while (deviceTimestamp < initTimestamp){updateTimestamp();}

    // Calculate delta time
    unsigned long currentTime = millis();
    deltaTime = (currentTime - previousTime) / 1000.0f;
    previousTime = currentTime;

    switchToFlexSensor4();
    delay(5);

    // Read flex sensors
    float rawFlex[5] = {
        (float)analogRead(flexSensor1),
        (float)analogRead(flexSensor2),
        (float)analogRead(flexSensor3),
        (float)analogRead(flexSensor4_5),
    };

    switchToFlexSensor5();
    delay(5);

    // Read flex sensor 5
    rawFlex[4] = analogRead(flexSensor4_5);

    // Filter flex sensor values using Kalman Filter
    float filteredFlex[5];
    for(int i = 0; i < 5; i++) {
        filteredFlex[i] = kalmanFilterFlex(flexKalman[i], rawFlex[i], prevFlexValues[i]);
    }

    // Set default magnetometer values (zeros)
    float magX = 0;  
    float magY = 0;  
    float magZ = 0;  

    // Read IMU data
    float accelX = myIMU.readFloatAccelX();
    float accelY = myIMU.readFloatAccelY();
    float accelZ = myIMU.readFloatAccelZ();
    float gyroX = myIMU.readFloatGyroX();
    float gyroY = myIMU.readFloatGyroY();
    float gyroZ = myIMU.readFloatGyroZ();

    // Calculate accelerometer angles
    float accelRoll, accelPitch;
    calculateAccelAngles(accelX, accelY, accelZ, accelRoll, accelPitch);

    // Apply complementary filter
    compRoll = alpha * (compRoll + gyroX * deltaTime) + (1 - alpha) * accelRoll;
    compPitch = alpha * (compPitch + gyroY * deltaTime) + (1 - alpha) * accelPitch;

    // Apply Kalman filter
    kalmanFilter(kalmanRoll, kalmanUncertaintyRoll, gyroX, accelRoll);
    kalmanFilter(kalmanPitch, kalmanUncertaintyPitch, gyroY, accelPitch);

    // Update Madgwick filter with IMU data only (no magnetometer)
    filter.updateIMU(gyroX, gyroY, gyroZ, accelX, accelY, accelZ);
              
    // Get orientation from Madgwick filter
    float madgwickYaw = filter.getYaw();

    updateTimestamp();

    // Combine the sensor values into an array for transmission
    float sensorValues[15] = {
        filteredFlex[0], filteredFlex[1], filteredFlex[2], filteredFlex[3], filteredFlex[4],
        accelX, accelY, accelZ, gyroX, gyroY, gyroZ,
        kalmanRoll, kalmanPitch, madgwickYaw, (float)deviceTimestamp,
    };

    // Create data string with all sensor data and filtered angles (removed mag data)
    String sensorData = "FilteredFlex1:" + String(filteredFlex[0], 2) + "," +
                      // "FilteredFlex2:" + String(filteredFlex[1], 2) + "," +
                      // "FilteredFlex3:" + String(filteredFlex[2], 2) + "," +
                      // "FilteredFlex4:" + String(filteredFlex[3], 2) + "," +
                      // "FilteredFlex5:" + String(filteredFlex[4], 2) + "," +
                      "KalmanRoll:" + String(kalmanRoll, 4) + "," +
                      "KalmanPitch:" + String(kalmanPitch, 4) + "," +
                      "MadgwickYaw:" + String(madgwickYaw, 4);
              
    // Print data to Serial Monitor
    Serial.println(sensorData);

    // Ensure Data is sent every specified interval
    while ((millis() - previousTime) < sendEvery) {}

    // Send the sensor values via the notify function
    randomCharacteristic.notify((uint8_t*)sensorValues, sizeof(sensorValues));
  }

  delay(1000);
}
#include<Wire.h>

const int MPU = 0x68; // I2C address of the MPU-6050

int nAccX, nAccY, nAccZ, nTimeStamp, nGyroX, nGyroY, nGyroZ;

void setup() {
  // setup acc sensor
  Wire.begin();
  
  Wire.beginTransmission(MPU);
  Wire.write(0x6B);  // PWR_MGMT_1 register
  Wire.write(0);     // set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);

  // setup serial
  Serial.begin(115200);
}

void loop() {
  Wire.beginTransmission(MPU);
  Wire.write(0x3B);  // starting with register 0x3B (ACCEL_XOUT_H)
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, 14, true); // request a total of 14 registers

  //  nTimeStamp = Wire.read() << 8 | Wire.read(); // 0x41 (TEMP_OUT_H) & 0x42 (TEMP_OUT_L)
 //  Serial.print(millis() ); Serial.print(", ");

  nAccX = Wire.read() << 8 | Wire.read(); // 0x3B (ACCEL_XOUT_H) & 0x3C (ACCEL_XOUT_L)
  nAccY = Wire.read() << 8 | Wire.read(); // 0x3D (ACCEL_YOUT_H) & 0x3E (ACCEL_YOUT_L)
  nAccZ = Wire.read() << 8 | Wire.read(); // 0x3F (ACCEL_ZOUT_H) & 0x40 (ACCEL_ZOUT_L)
  Serial.print(nAccX); Serial.print(", ");
  Serial.print(nAccY); Serial.print(", ");
  Serial.print(nAccZ); Serial.print(", ");

  nGyroX = Wire.read() << 8 | Wire.read(); // 0x43 (GYRO_XOUT_H) & 0x44 (GYRO_XOUT_L)
  nGyroY = Wire.read() << 8 | Wire.read(); // 0x45 (GYRO_YOUT_H) & 0x46 (GYRO_YOUT_L)
  nGyroZ = Wire.read() << 8 | Wire.read(); // 0x47 (GYRO_ZOUT_H) & 0x48 (GYRO_ZOUT_L)
  Serial.print(nGyroX);Serial.print(", ");
  Serial.print(nGyroY);Serial.print(", ");
  Serial.println(nGyroZ);
}

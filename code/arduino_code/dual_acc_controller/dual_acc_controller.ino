// This module read two MPU6050 accelerometer at the same time

#include<Wire.h>

#define MPU6050_I2C_ADDRESS_0 0x68
#define MPU6050_I2C_ADDRESS_1 0x69

#define MPU6050_ACCEL_CONFIG       0x1C   // R/W
#define MPU6050_PWR_MGMT_1         0x6B   // R/W

// Defines for the bits, to be able to change 
// between bit number and binary definition.
// By using the bit number, programming the sensor 
// is like programming the AVR microcontroller.
// But instead of using "(1<<X)", or "_BV(X)", 
// the Arduino "bit(X)" is used.
#define MPU6050_D0 0
#define MPU6050_D1 1
#define MPU6050_D2 2
#define MPU6050_D3 3
#define MPU6050_D4 4
#define MPU6050_D5 5
#define MPU6050_D6 6
#define MPU6050_D7 7

// ACCEL_CONFIG Register
// The XA_ST, YA_ST, ZA_ST are bits for selftest.
// The AFS_SEL sets the range for the accelerometer.
// These are the names for the bits.
// Use these only with the bit() macro.
#define MPU6050_ACCEL_HPF0 MPU6050_D0
#define MPU6050_ACCEL_HPF1 MPU6050_D1
#define MPU6050_ACCEL_HPF2 MPU6050_D2
#define MPU6050_AFS_SEL0   MPU6050_D3
#define MPU6050_AFS_SEL1   MPU6050_D4
#define MPU6050_ZA_ST      MPU6050_D5
#define MPU6050_YA_ST      MPU6050_D6
#define MPU6050_XA_ST      MPU6050_D7

// Combined definitions for the AFS_SEL values
#define MPU6050_AFS_SEL_0 (0)
#define MPU6050_AFS_SEL_1 (bit(MPU6050_AFS_SEL0))
#define MPU6050_AFS_SEL_2 (bit(MPU6050_AFS_SEL1))
#define MPU6050_AFS_SEL_3 (bit(MPU6050_AFS_SEL1)|bit(MPU6050_AFS_SEL0))

// Register names according to the datasheet.
// According to the InvenSense document 
// "MPU-6000 and MPU-6050 Register Map 
// and Descriptions Revision 3.2", there are no registers
// at 0x02 ... 0x18, but according other information 
// the registers in that unknown area are for gain 
// and offsets.
#define MPU6050_ACCEL_XOUT_H       0x3B   // R  

int nAccX0, nAccY0, nAccZ0, nGyroX0, nGyroY0, nGyroZ0;

int nAccX1, nAccY1, nAccZ1, nGyroX1, nGyroY1, nGyroZ1;

void setup() {
	
  Wire.begin();
  
  // weak up sensor 0
  Wire.beginTransmission(MPU6050_I2C_ADDRESS_0);
  Wire.write(MPU6050_PWR_MGMT_1);  // PWR_MGMT_1 register
  Wire.write(0);     // set to zero (wakes up the MPU6050_I2C_ADDRESS_0-6050)
  Wire.endTransmission(true);
  
  // set range of sensor 0
  Wire.beginTransmission(MPU6050_I2C_ADDRESS_0);
  Wire.write(MPU6050_ACCEL_CONFIG);
  Wire.write(MPU6050_AFS_SEL_3);     // set range
  Wire.endTransmission(true);
  
  // weak up sensor 1
  Wire.beginTransmission(MPU6050_I2C_ADDRESS_1);
  Wire.write(MPU6050_PWR_MGMT_1);  // PWR_MGMT_1 register
  Wire.write(0);     // set to zero (wakes up the MPU6050_I2C_ADDRESS_0-6050)
  Wire.endTransmission(true);
  
  // set range of sensor 1
  Wire.beginTransmission(MPU6050_I2C_ADDRESS_1);
  Wire.write(MPU6050_ACCEL_CONFIG);
  Wire.write(MPU6050_AFS_SEL_3);     // set range
  Wire.endTransmission(true);

  // setup serial
  Serial.begin(115200);
}

void loop() {

	/*----read sensor 0----*/
	Wire.beginTransmission(MPU6050_I2C_ADDRESS_0);
	Wire.write(MPU6050_ACCEL_XOUT_H);  // starting with register 0x3B (ACCEL_XOUT_H)
	Wire.endTransmission(false);
	Wire.requestFrom(MPU6050_I2C_ADDRESS_0, 14, true); // request a total of 14 registers

	nAccX0 = Wire.read() << 8 | Wire.read(); // 0x3B (ACCEL_XOUT_H) & 0x3C (ACCEL_XOUT_L)
	nAccY0 = Wire.read() << 8 | Wire.read(); // 0x3D (ACCEL_YOUT_H) & 0x3E (ACCEL_YOUT_L)
	nAccZ0 = Wire.read() << 8 | Wire.read(); // 0x3F (ACCEL_ZOUT_H) & 0x40 (ACCEL_ZOUT_L)
	nGyroX0 = Wire.read() << 8 | Wire.read(); // 0x43 (GYRO_XOUT_H) & 0x44 (GYRO_XOUT_L)
	nGyroY0 = Wire.read() << 8 | Wire.read(); // 0x45 (GYRO_YOUT_H) & 0x46 (GYRO_YOUT_L)
	nGyroZ0 = Wire.read() << 8 | Wire.read(); // 0x47 (GYRO_ZOUT_H) & 0x48 (GYRO_ZOUT_L)
	
	/*----read sensor 1----*/
	Wire.beginTransmission(MPU6050_I2C_ADDRESS_1);
	Wire.write(MPU6050_ACCEL_XOUT_H);  // starting with register 0x3B (ACCEL_XOUT_H)
	Wire.endTransmission(false);
	Wire.requestFrom(MPU6050_I2C_ADDRESS_1, 14, true); // request a total of 14 registers

	nAccX1 = Wire.read() << 8 | Wire.read(); // 0x3B (ACCEL_XOUT_H) & 0x3C (ACCEL_XOUT_L)
	nAccY1 = Wire.read() << 8 | Wire.read(); // 0x3D (ACCEL_YOUT_H) & 0x3E (ACCEL_YOUT_L)
	nAccZ1 = Wire.read() << 8 | Wire.read(); // 0x3F (ACCEL_ZOUT_H) & 0x40 (ACCEL_ZOUT_L)

	nGyroX1 = Wire.read() << 8 | Wire.read(); // 0x43 (GYRO_XOUT_H) & 0x44 (GYRO_XOUT_L)
	nGyroY1 = Wire.read() << 8 | Wire.read(); // 0x45 (GYRO_YOUT_H) & 0x46 (GYRO_YOUT_L)
	nGyroZ1 = Wire.read() << 8 | Wire.read(); // 0x47 (GYRO_ZOUT_H) & 0x48 (GYRO_ZOUT_L)

	/*----print to serial----*/
	// Serial.print(millis() ); Serial.print(", ");
	// sensor0
	Serial.print(nAccX0); Serial.print(", ");
	Serial.print(nAccY0); Serial.print(", ");
	Serial.print(nAccZ0); Serial.print(", ");

	Serial.print(nGyroX0); Serial.print(", ");
	Serial.print(nGyroY0); Serial.print(", ");
	Serial.print(nGyroZ0); Serial.print(", ");

	// sensor 1
	Serial.print(nAccX1); Serial.print(", ");
	Serial.print(nAccY1); Serial.print(", ");
	Serial.print(nAccZ1); Serial.print(", ");

	Serial.print(nGyroX1); Serial.print(", ");
	Serial.print(nGyroY1); Serial.print(", ");
	Serial.println(nGyroZ1);
}

#include <stdio.h>

const int ANALOG_OUT_PIN = 3;			// the PWM PIN
const int VIBRATION_DURATION = 1400; 	// the duration of each vibration segment
const int INTERVAL_DURATION = 0; 		// static duration btw vibration segments
const int RESET_DURATION = 1000;		// the time interval btw trials
const int SERIAL_BAUD_RATE = 9600;		// baud rate of serial
const int SAFE_VOLTAGE = 150;			// max voltage
const int VOLTAGE_STEP = 10;				// the voltage to increase each time
const int SWEEP_STARTING_VOLTAGE = 30;	// the starting voltage of voltage sweeping
const int SWEEP_END_VOLTAGE = 150;       // the end voltage of voltage sweeping

bool g_bSweep = false;						// indicator of sweeping state
int g_nCurrentVoltage = 0;					// current voltage
char g_arrStrFormatBuffer [50] = {'\0'};	// buffer for output
int g_nTrailNum = 10; 						// number of trials to perform
int g_nTrialCount = 0; 						// number of trials conducted

void setup() 
{
  Serial.begin(SERIAL_BAUD_RATE); // opens serial port, sets data rate to 9600 bps
}

void loop() 
{

  // response to user input
  if (Serial.available() > 0) 
  {
    int nCmd = Serial.parseInt();

    if (nCmd == 0) // 0: stop testing
    {
      setVibration(0);
	  clearState();
	  
	  Serial.println("Test is stopped.");
    }
    else if (nCmd > 0) // vibrate as user input for a single segment
    {
	  if(g_bSweep == true)
	  {
		  Serial.println("Invalid input: the sweeping test is running, stop it first!");
		  return;
	  }
	  
	  sprintf(g_arrStrFormatBuffer, "Start single vibration, voltage:%d.", nCmd);
      Serial.println(g_arrStrFormatBuffer);

      // vibrate
	  delay(INTERVAL_DURATION);
      setVibration(nCmd);
      delay(VIBRATION_DURATION);
      setVibration(0);
    }
    else // < 0, sweep for nCmd times
    {
	  g_nTrailNum = -1 * (nCmd);
	  g_nTrialCount = 0;
      g_nCurrentVoltage = SWEEP_STARTING_VOLTAGE;
      g_bSweep = true;
      sprintf(g_arrStrFormatBuffer, "Start %d trials...", g_nTrailNum);
	  Serial.println(g_arrStrFormatBuffer);
      delay(RESET_DURATION);
    }
  }

  // conducting test
  if (g_bSweep == true)
  {
	if(g_nTrialCount >= g_nTrailNum) // all trials are done
	{
		clearState();
		setVibration(0);
		delay(INTERVAL_DURATION);
		
		Serial.println("All trials are finished.");
	}
	  
    if (g_nCurrentVoltage <= SWEEP_END_VOLTAGE)
    {
		// vibrate
		setVibration(g_nCurrentVoltage);
		delay(VIBRATION_DURATION);

		// // stop for a while
		// setVibration(0);
		// delay(INTERVAL_DURATION);

		g_nCurrentVoltage += VOLTAGE_STEP;
    }
    else
    {
		sprintf(g_arrStrFormatBuffer, "-->Trial #%d is finished.", g_nTrialCount);
		Serial.println(g_arrStrFormatBuffer);
		++g_nTrialCount;
		
		// stay static for a while
		setVibration(0);
		delay(RESET_DURATION);
		
		g_nCurrentVoltage = SWEEP_STARTING_VOLTAGE;
    }
  }
}

/*
 * This function set the vibration intensity via PWM
 */
void setVibration(int nIntensity)
{
  int nVol = min(nIntensity, SAFE_VOLTAGE);
  analogWrite(ANALOG_OUT_PIN,  nVol);

  sprintf(g_arrStrFormatBuffer, "current intensity: %d.", nVol);
  Serial.println(g_arrStrFormatBuffer);
}

/*
 * This function clear the test state
 */
void clearState()
{
	g_nCurrentVoltage = 0;
	g_nTrialCount = 0;
	g_bSweep = false;
}
